package net.forlindon.dynamic.objectoriented.neural.networks.knot;

public class ReluKnot extends Knot {

    public ReluKnot(int id) {
        super(id);
    }

    @Override
    public double activation(double d) {
        return Math.max(d,0);
    }

    @Override
    public double derivative(double d) {
        return Math.max(d,0);
    }
}
