package net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class TanhTensor extends SimpleActivationTensor {

    @Override
    public void activate(Tensor... args) {
        super.activate(args);
        this.val = (Math.exp(this.val) - Math.exp(-this.val))/(Math.exp(this.val) + Math.exp(-this.val));
    }

    @Override
    public void derivative(Tensor... args) {
        this.grad*=1-Math.pow(this.val,2);
        super.derivative(args);
    }

    @Override
    public Tensor copy() {
        return new TanhTensor();
    }
}
