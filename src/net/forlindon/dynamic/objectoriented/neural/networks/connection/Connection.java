package net.forlindon.dynamic.objectoriented.neural.networks.connection;

import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

public abstract class Connection {

    Knot src;
    Knot dest;
    double w;
    double grad;

    public Connection(Knot src,Knot dest) {
        this.src = src;
        this.dest = dest;
        this.w = Math.random()-0.5;
    }

    public double grad() {
        return this.grad;
    }

    public double weight() {
        return this.w;
    }

    public void ff() {
        dest.push(this.w*src.value() + dest.bias());
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof Connection c && this.dest == c.dest;
    }
}
