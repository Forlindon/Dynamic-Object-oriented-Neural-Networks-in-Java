package net.forlindon.dynamic.objectoriented.neural.networks.tensor;

public abstract class Tensor {

    public double val;
    public double grad;

    public Tensor() {
        this(0,0);
    }

    public Tensor(double v) {
        this(v,0);
    }

    public Tensor(double v, double grad) {
        this.val = v;
        this.grad = grad;
    }

    public abstract void activate(Tensor... args);

    public abstract void derivative(Tensor... args);

    public void push(double d) {
        this.val += d;
    }

    public void pushGrad(double d) {
        this.grad += d;
    }

    @Override
    public String toString() {
        return String.format("Tensor{Value: %.2g, Grad: %.2g}", this.val, this.grad);
    }
}
