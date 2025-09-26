package net.forlindon.dynamic.objectoriented.neural.networks.knot;

public class TanhKnot extends Knot {

    public TanhKnot(int id) {
        super(id);
    }

    @Override
    public double activation(double d) {
        return (Math.exp(d) - Math.exp(-d)) / (Math.exp(d) + Math.exp(-d));
    }

    @Override
    public double derivative(double d) {
        return 1.0-Math.pow(d,2);
    }
}
