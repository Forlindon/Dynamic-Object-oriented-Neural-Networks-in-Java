package net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class SimpleActivationTensor extends Tensor {

    @Override
    public void activate(Tensor... args) {
        this.push(args[0].val + args[1].val);
    }

    @Override
    public void derivative(Tensor... args) {
        args[0].pushGrad(this.grad);
        args[1].pushGrad(this.grad);
    }
}
