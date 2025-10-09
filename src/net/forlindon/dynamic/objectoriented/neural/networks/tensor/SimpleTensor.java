package net.forlindon.dynamic.objectoriented.neural.networks.tensor;

public class SimpleTensor extends Tensor {

    public SimpleTensor(double v) {
        this(v,0);
    }

    public SimpleTensor(double v, double grad) {
        super(v);
        this.grad = grad;
    }


    @Override
    public void activate(Tensor... args) {
        for (Tensor t : args) {
            this.push(t.val);
        }
    }

    @Override
    public void derivative(Tensor... args) {
        for (Tensor t : args) {
            t.pushGrad(this.grad);
        }
    }

    @Override
    public Tensor copy() {
        return new SimpleTensor(this.val,this.grad);
    }
}
