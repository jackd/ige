from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import gin


def _as_built_layer(fn, input_shape):
    if fn is None:
        return None
    elif isinstance(fn, tf.keras.layers.Layer):
        fn.build(input_shape)
        return fn
    else:
        inputs = [tf.keras.layers.Input(shape[1:]) for shape in input_shape]
        outputs = fn(inputs)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)


class UnrolledOptimizer(tf.keras.layers.Layer):
    def __init__(
            self, energy_fn, num_steps, num_optimized=None,
            map_fn=None, loop_kwargs=None, dtype=tf.float32,
            **kwargs):
        self.energy_fn = energy_fn
        self.num_steps = num_steps
        self.num_optimized = num_optimized
        self.map_fn = map_fn
        self.loop_kwargs = loop_kwargs
        super(UnrolledOptimizer, self).__init__(dtype=dtype, **kwargs)
    
    def get_config(self):
        return dict(
            energy_fn=tf.keras.utils.serialize_keras_object(self.energy_fn),
            num_steps=self.num_steps,
            num_optimized=self.num_optimized,
            map_layer=tf.keras.utils.serialize_keras_object(self.map_fn),
            loop_kwargs=self.loop_kwargs,
        )

    def build(self, input_shape):
        if self.built:
            return
        self.energy_layer = _as_built_layer(self.energy_fn, input_shape)
        self.map_layer = _as_built_layer(
            self.map_fn,
            input_shape if self.num_optimized is None else
                input_shape[:self.num_optimized])
        super(UnrolledOptimizer, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        shape = input_shape
        if self.num_optimized is None:
            if isinstance(shape, tf.TensorShape):
                opt_shape = [shape]
            else:
                opt_shape = shape
        else:
            opt_shape = shape[:self.num_optimized]
        if self.map_layer is None:
            out_shape = opt_shape
        else:
            out_shape = self.map_layer.compute_output_shape(opt_shape)
            if isinstance(out_shape, tf.TensorShape):
                out_shape = [out_shape]
        out_shape = [
            tf.TensorShape([self.num_steps] + s.as_list()) for s in out_shape]
        return opt_shape + out_shape
    
    @abc.abstractmethod
    def initial_state(self, opt_args):
        raise NotImplementedError
    
    @abc.abstractmethod
    def step(self, opt_args, rest_args, state):
        raise NotImplementedError

    def call(self, inputs):
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        elif not isinstance(inputs, (list)):
            inputs = [inputs]
        self.build([i.shape for i in inputs])

        if self.num_optimized is None:
            opt_args = inputs
            rest = []
        else:
            if isinstance(inputs, tf.Tensor):
                inputs = [inputs]
            opt_args = inputs[:self.num_optimized]
            rest = inputs[self.num_optimized:]
        dtype = self.dtype if self.map_layer is None else self.map_layer.dtype 
        
        initial_state = self.initial_state(opt_args)
        shapes = self.compute_output_shape([i.shape for i in inputs])
        assert(isinstance(shapes, list))

        out_arr = [tf.TensorArray(
            size=self.num_steps, element_shape=s.shape, dtype=dtype)
                   for s in opt_args]
        
        def body(i, opt_args, rest, state, out_arr):
            opt_args, state = self.step(opt_args, rest, state)
            if self.map_layer is not None:
                out = self.map_layer(opt_args)
            else:
                out = opt_args
            out_arr = [oa.write(i, o) for o, oa in zip(out, out_arr)]
            return i+1, opt_args, rest, state, out_arr

        def cond(i, opt_ags, rest, state, out_arr):
            return i < self.num_steps

        loop_kwargs = self.loop_kwargs or {}
        n, opt_args, rest, state, out_arr = tf.while_loop(
            cond, body, [0, opt_args, rest, initial_state, out_arr],
            **loop_kwargs)

        del n, state
        out_vals = [o.stack() for o in out_arr]
        for out_val in out_vals:
            out_val.set_shape([self.num_steps] + out_val.shape[1:].as_list())
        return opt_args + out_vals


@gin.configurable(blacklist=['energy_fn', 'num_optimized'])
class UnrolledSGD(UnrolledOptimizer):
    def __init__(
            self, energy_fn, num_optimized, num_steps=4,
            learning_rate=1.0, momentum=None, gradient_clip_value=None,
            learn_learning_rate=False, learn_momentum=False,
            learn_gradient_clip_value=False, take_momentum_abs=True,
            **kwargs):
        super(UnrolledSGD, self).__init__(
            energy_fn=energy_fn,
            num_steps=num_steps,
            num_optimized=num_optimized,
            **kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.learn_learning_rate = learn_learning_rate
        self.learn_momentum = learn_momentum
        self.gradient_clip_value = gradient_clip_value
        self.learn_gradient_clip_value = learn_gradient_clip_value
        self.take_momentum_abs = take_momentum_abs
    
    def get_config(self):
        config = super(UnrolledSGD, self).get_config()
        config.update(dict(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            learn_learning_rate=self.learn_learning_rate,
            learn_momentum=self.learn_momentum,
            gradient_clip_value=self.gradient_clip_value,
            learn_gradient_clip_value=self.learn_gradient_clip_value,
            take_momentum_abs=self.take_momentum_abs,
        ))
        return config
    
    def build(self, input_shape):
        if self.built:
            return
        dtype = self.dtype
        if self.learn_learning_rate:
            self.learning_rate_used = self.add_weight(
                'learning_rate',
                shape=(),
                initializer=tf.keras.initializers.constant(
                    self.learning_rate, dtype=dtype))
        else:
            self.learning_rate_used = tf.convert_to_tensor(
                self.learning_rate, dtype=dtype)

        if self.learn_momentum:
            self.momentum_used = self.add_weight(
                'momentum',
                shape=(),
                initializer=tf.keras.initializers.constant(
                    self.momentum, dtype=dtype))
            if self.take_momentum_abs:
                self.momentum_used = tf.abs(self.momentum_used)
        elif self.momentum:
            self.momentum_used = tf.convert_to_tensor(
                self.momentum, dtype=dtype)
        else:
            self.momentum_used = None
        
        if self.learn_gradient_clip_value:
            self.gradient_clip_value_used = self.add_weight(
                'gradient_clip_value',
                shape=(),
                initializer=tf.keras.initializers.constant(
                    self.gradient_clip_value, dtype=dtype))
        elif self.gradient_clip_value:
            self.gradient_clip_value_used = tf.convert_to_tensor(
                self.gradient_clip_value, dtype=dtype)
        else:
            self.gradient_clip_value_used = None
        super(UnrolledSGD, self).build(input_shape)
    
    def initial_state(self, opt_args):
        if self.momentum_used is None:
            return []
        else:
            return [tf.zeros_like(a) for a in opt_args]
    
    def step(self, opt_args, rest_args, state):
        with tf.GradientTape() as tape:
            tape.watch(opt_args)
            energy = self.energy_layer(opt_args + rest_args)
        gradients = tape.gradient(energy, opt_args)
        if self.gradient_clip_value_used is not None:
            gradients = [
                tf.clip_by_value(
                    g,
                    -self.gradient_clip_value_used, # pylint: disable=invalid-unary-operand-type
                    self.gradient_clip_value_used)
                for g in gradients]
        if self.momentum_used is None:
            opt_args = [
                a - self.learning_rate_used*g
                for a, g in zip(opt_args, gradients)]
        else:
            state = [
                self.momentum_used * s + g for s, g in zip(state, gradients)]
            opt_args = [
                a - self.learning_rate_used * s
                for a, s in zip(opt_args, state)]

        return opt_args, state