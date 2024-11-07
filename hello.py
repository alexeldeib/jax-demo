import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import *

import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from flax import nnx
import orbax.checkpoint as ocp

import optax

print(f'You have 8 “fake” JAX devices now: {jax.devices()}')

# create device mesh
mesh = Mesh(devices=np.array(jax.devices()).reshape(2, 4),
            axis_names=('data', 'model'))

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

path = "/tmp/relu"
options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2, create=True)
mngr = ocp.CheckpointManager(path, options=options)

# create model for ckpt restore
with mesh:
    sharded_model = create_sharded_model()

latest_step = 0
if mngr.latest_step() is not None:
    print(f"found checkpoint at step {mngr.latest_step()}, restoring")
    latest_step = mngr.latest_step()
    graphdef, state = nnx.split(sharded_model)
    loaded_shard = mngr.restore(
        latest_step,
        args=ocp.args.StandardRestore(state),
    )
    sharded_model = nnx.merge(graphdef, loaded_shard)
    print("restored checkpoint")

data_sharding = NamedSharding(mesh, PartitionSpec('data', None))

## training setup
optimizer = nnx.Optimizer(sharded_model, optax.adam(1e-3))  # reference sharing

# data transfer
input = jax.device_put(jax.random.normal(jax.random.key(1), (8, 1024)), data_sharding)
label = jax.device_put(jax.random.normal(jax.random.key(2), (8, 1024)), data_sharding)

with mesh:
    for i in range(5):
        loss = train_step(sharded_model, optimizer, input, label)
        print(loss)    # Model (over-)fitting to the labels quickly
        mngr.save(latest_step, args=ocp.args.StandardSave(nnx.state(sharded_model)))
        latest_step += 1

mngr.wait_until_finished()

print(f"training finished at step {mngr.latest_step()}")
print(nnx.state(sharded_model))
