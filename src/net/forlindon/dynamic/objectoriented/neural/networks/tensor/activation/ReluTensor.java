package net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class ReluTensor extends SimpleActivationTensor {

    @Override
    public void activate(Tensor... args) {
        super.activate(args);
        this.val = Math.max(this.val,0);
    }

    @Override
    public void derivative(Tensor... args) {
        this.grad*=(args[0].val+args[1].val) > 0 ? 1 : 0;
        super.derivative(args);
    }

    @Override
    public Tensor copy() {
        return new ReluTensor();
    }
}
