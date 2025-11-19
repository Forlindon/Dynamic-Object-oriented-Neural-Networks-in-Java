package net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class ParametricRelu extends SimpleActivationTensor {

    double alpha;

    public ParametricRelu(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public void activate(Tensor... args) {
        super.activate(args);
        this.val = this.val > 0 ? this.val : this.val * this.alpha;
    }

    @Override
    public void derivative(Tensor... args) {
        double x = (args[0].val+args[1].val);
        this.grad*= x > 0 ? 1 : this.alpha;
        super.derivative(args);
    }

    @Override
    public Tensor copy() {
        return new ParametricRelu(this.alpha);
    }

    @Override
    public String toString() {
        return String.format("{v: %.2g, g: %.2g, a: %.2g}", this.val, this.grad, this.alpha);
    }

    @Override
    public void adjust(double d) {
        this.alpha -= d*this.grad;
        this.grad = 0;
    }
}
