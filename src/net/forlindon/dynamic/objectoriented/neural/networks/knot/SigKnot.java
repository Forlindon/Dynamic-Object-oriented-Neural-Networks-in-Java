package net.forlindon.dynamic.objectoriented.neural.networks.knot;

public class SigKnot extends Knot {

    public SigKnot(int id) {
        super(id);
    }

    @Override
    public double activation(double d) {
        return 1.0/(1.0+Math.pow(Math.E, -d));
    }

    @Override
    public double derivative(double d) {
        return d*(1-d);
    }
}
