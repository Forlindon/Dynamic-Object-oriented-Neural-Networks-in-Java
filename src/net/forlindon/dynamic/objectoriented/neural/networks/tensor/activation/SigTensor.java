package net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class SigTensor extends SimpleActivationTensor {

    @Override
    public void activate(Tensor... args) {
        super.activate(args);
        this.val = sig(this.val);
    }

    @Override
    public void derivative(Tensor... args) {
        this.grad*=(this.val*(1-this.val));
        super.derivative(args);
    }

    private double sig(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public Tensor copy() {
        return new SigTensor();
    }
}
