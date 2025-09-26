package net.forlindon.dynamic.objectoriented.neural.networks.tensor;

public class SimpleTensor extends Tensor {

    public SimpleTensor(double v) {
        super(v);
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
}
