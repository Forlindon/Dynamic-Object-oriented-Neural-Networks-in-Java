package net.forlindon.dynamic.objectoriented.neural.networks.tensor;

public class MulTensor extends SimpleTensor {

    public MulTensor() {
        super(0);
    }

    public MulTensor(double v) {
        super(v);
    }

    public MulTensor(double v, double grad) {
        super(v, grad);
    }

    @Override
    public void activate(Tensor... args) {
        this.push(args[0].val * args[1].val);
    }

    @Override
    public void derivative(Tensor... args) {
        args[0].pushGrad(this.grad*args[1].val);
        args[1].pushGrad(this.grad*args[0].val);
    }

    @Override
    public Tensor copy() {
        return new MulTensor(this.val,this.grad);
    }
}
