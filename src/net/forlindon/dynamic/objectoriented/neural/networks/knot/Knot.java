package net.forlindon.dynamic.objectoriented.neural.networks.knot;

import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.MulTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.SimpleTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

public abstract class Knot extends MulTensor {

    private final List<Connection> OUTBOUND;
    private final int LAYER_ID;
    private Tensor bias = new SimpleTensor(Math.random() * 0.5);

    public Knot(int id) {
        this.LAYER_ID = id;
        this.OUTBOUND = new ArrayList<>();
    }

    public void connect(Knot other, BiFunction<Knot, Knot, Connection> factory) {
        if (other.LAYER_ID <= this.LAYER_ID) throw new IllegalArgumentException("Invalid LAYER_ID");
        Connection c = factory.apply(this, other);
        if (this.OUTBOUND.contains(c)) throw new IllegalArgumentException("No duplicate connections");
        this.OUTBOUND.add(c);
    }

    @Override
    public void derivative(Tensor... args) {
        super.derivative(args);
        this.bias.pushGrad(this.grad);
    }

    public abstract double activation(double d);

    public void pop() {
        this.val = activation(this.val) + this.bias.val;
        for (Connection c : this.OUTBOUND) {
            c.ff();
        }
    }

    public void backward() {
        for (Connection c : this.OUTBOUND) {
            c.fb();
        }
    }

    public void push(double d) {
        this.val += d;
    }

    public double value() {
        return this.val;
    }

    public void reset() {
        this.val = 0.0;
    }

    @Override
    public String toString() {
        return String.format( "%s{Value: %.2g, Bias: %.2g, Grad: %.2g, Connections: %d}", Knot.class.getSimpleName(), this.val, this.bias.val, this.grad, this.OUTBOUND.size());
    }

    public double bias() {
        return bias.val;
    }

    public int id() {
        return this.LAYER_ID;
    }
}
