package net.forlindon.dynamic.objectoriented.neural.networks.tensor;

public class AddTensor extends SimpleTensor {

    public AddTensor(double v) {
        super(v);
    }

    public AddTensor(double v, double grad) {
        super(v, grad);
    }

    @Override
    public void activate(Tensor... args) {
        this.push(args[0].val+args[1].val);
    }

    @Override
    public void derivative(Tensor... args) {
        args[0].pushGrad(this.grad);
        args[1].pushGrad(this.grad);
    }

    @Override
    public Tensor copy() {
        return new AddTensor(this.val, this.grad);
    }
}
