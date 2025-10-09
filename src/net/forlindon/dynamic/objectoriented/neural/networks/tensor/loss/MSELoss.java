package net.forlindon.dynamic.objectoriented.neural.networks.tensor.loss;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class MSELoss extends Tensor {

    @Override
    public void activate(Tensor... args) {
        this.val = Math.pow(args[0].val-args[1].val,2);
    }

    @Override
    public void derivative(Tensor... args) {
        this.grad = 2*(args[0].val-args[1].val);
        args[0].pushGrad(this.grad);
        args[1].pushGrad(-this.grad);
    }

    @Override
    public Tensor copy() {
        return new MSELoss();
    }
}
