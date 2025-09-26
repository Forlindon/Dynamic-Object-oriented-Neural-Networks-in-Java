package net.forlindon.dynamic.objectoriented.neural.networks.knot;

public class BaseKnot extends Knot{
    public BaseKnot(int id) {
        super(id);
    }

    @Override
    public double activation(double d) {
        return d;
    }
}
