import os
import random
import time

from typing import *

import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from flax import nnx
import orbax.checkpoint as ocp
import optax

class DotReluDot(nnx.Module):
    def __init__(self, depth: int, rngs: nnx.Rngs):
        init_fn = nnx.initializers.lecun_normal()

        # Initialize a sublayer `self.dot1` and annotate its kernel with.
        # `sharding (None, 'model')`.
        self.dot1 = nnx.Linear(
            depth, depth,
            kernel_init=nnx.with_partitioning(init_fn, (None, 'model')),
            use_bias=False,  # or use `bias_init` to give it annotation too
            rngs=rngs)

        # Initialize a weight param `w2` and annotate with sharding ('model', None).
        # Note that this is simply adding `.sharding` to the variable as metadata!
        self.w2 = nnx.Param(
            init_fn(rngs.params(), (depth, depth)),  # RNG key and shape for W2 creation
            sharding=('model', None),
        )

    def __call__(self, x: jax.Array):
        y = self.dot1(x)
        y = jax.nn.relu(y)
        # In data parallelism, input / intermediate value's first dimension (batch)
        # will be sharded on `data` axis
        y = jax.lax.with_sharding_constraint(y, PartitionSpec('data', 'model'))
        z = jnp.dot(y, self.w2.value)
        return z
    
@nnx.jit
def create_sharded_model():
    model = DotReluDot(1024, rngs=nnx.Rngs(0)) # Unsharded at this moment.
    state = nnx.state(model)                   # The model's state, a pure pytree.
    pspecs = nnx.get_partition_spec(state)     # Strip out the annotations from state.
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)           # The model is sharded now!
    return model


@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model: DotReluDot):
        y_pred = model(x)
        return jnp.mean((y_pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)

    return loss


random.seed()

idx = int(os.environ["JOB_COMPLETION_INDEX"])
num_processes = int(os.environ["JOB_SIZE"])
jax.distributed.initialize(
    coordinator_address=os.environ["COORDINATOR_ADDRESS"],
    num_processes=num_processes,
    process_id=idx,
)

# create device mesh
mesh_dim = int(num_processes * 8 / 2)
mesh = Mesh(
    devices=np.array(jax.devices()).reshape(2, mesh_dim),
    axis_names=('data', 'model'),
)

with mesh:
    sharded_model = create_sharded_model()

path = "/storage/relu"
options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2, create=True)
mngr = ocp.CheckpointManager(path, options=options)
latest_step = 0

if mngr.latest_step() is not None:
    print(f"found checkpoint at step {mngr.latest_step()}, restoring", flush=True)
    latest_step = mngr.latest_step()
    graphdef, state = nnx.split(sharded_model)
    loaded_shard = mngr.restore(
        latest_step,
        args=ocp.args.StandardRestore(state),
    )
    sharded_model = nnx.merge(graphdef, loaded_shard)
    print("restored checkpoint", flush=True)

data_sharding = NamedSharding(mesh, PartitionSpec('data', None))

## training setup
optimizer = nnx.Optimizer(sharded_model, optax.adam(1e-3))  # reference sharing

# data transfer
input = jax.device_put(jax.random.normal(jax.random.key(1), (8, 1024)), data_sharding)
label = jax.device_put(jax.random.normal(jax.random.key(2), (8, 1024)), data_sharding)

with mesh:
    while latest_step <= 30:
        loss = train_step(sharded_model, optimizer, input, label)
        print(f"step: {latest_step}  loss: {loss}", flush=True)    # Model (over-)fitting to the labels quickly
        mngr.save(latest_step, args=ocp.args.StandardSave(nnx.state(sharded_model)))
        latest_step += 1
        if random.randrange(10) == 0:
            mngr.wait_until_finished()
            print("simulated failure", flush=True)
            exit(1)

mngr.wait_until_finished()

print(f"training finished at step {mngr.latest_step()}", flush=True)
