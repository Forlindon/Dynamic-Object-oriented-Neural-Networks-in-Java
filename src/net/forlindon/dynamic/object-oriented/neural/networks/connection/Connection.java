package connection;

import knot.Knot;

public abstract class Connection {

    Knot dest;
    double w;

    public Connection(Knot knot) {
        this.dest = knot;
        this.w = Math.random()-0.5;
    }

    public void ff(double d) {
        dest.push(this.w*d + dest.bias());
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof Connection c && this.dest == c.dest;
    }
}
