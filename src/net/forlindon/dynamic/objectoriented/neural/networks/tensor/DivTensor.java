package net.forlindon.dynamic.objectoriented.neural.networks.tensor;

public class DivTensor extends SimpleTensor {
    public DivTensor(double v) {
        super(v);
    }

    public DivTensor(double v, double grad) {
        super(v, grad);
    }

    @Override
    public void activate(Tensor... args) {
        this.push(args[0].val/args[1].val);
    }

    @Override
    public void derivative(Tensor... args) {
        args[0].pushGrad(this.grad * (1.0/args[1].val));
        args[1].pushGrad(this.grad * args[0].val);
    }

    @Override
    public Tensor copy() {
        return new DivTensor(this.val,this.grad);
    }
}
