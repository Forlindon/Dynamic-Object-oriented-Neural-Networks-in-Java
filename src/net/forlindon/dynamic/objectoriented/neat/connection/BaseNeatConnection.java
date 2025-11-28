package net.forlindon.dynamic.objectoriented.neat.connection;

import net.forlindon.dynamic.objectoriented.neat.genetic.Genotype;
import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neat.genetic.MutationFactory;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neural.networks.connection.BaseConnection;
import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

import java.util.List;
import java.util.function.BiFunction;

public class BaseNeatConnection extends BaseConnection implements Genotype {

    private final int inovNum;
    private boolean isActive;

    public BaseNeatConnection(InnovationSource innovationSource, Knot src, Knot dest) {
        super(src, dest);
        this.inovNum = innovationSource.getNext();
        this.isActive = true;
    }

    protected BaseNeatConnection(int inovNum, Knot src, Knot dest, double val, double grad, boolean active) {
        super(src, dest);
        this.inovNum = inovNum;
        this.val = val;
        this.grad = grad;
        this.isActive = active;
    }

    @Override
    public int inov() {
        return this.inovNum;
    }

    @Override
    public void mutate() {
        this.push(MutationFactory.N()*0.1);
        if (Math.random() < 0.3) this.val = MutationFactory.N();
        if (Math.random() < 0.1) this.isActive = !this.isActive;
    }

    @Override
    public void ff() {
        if (isActive()) super.ff();
    }

    @Override
    public double getMutationRate() {
        return 0.9;
    }

    @Override
    public String toString() {
        return Integer.toHexString(System.identityHashCode(dest()));
        // return String.format("src: %s, dest: %s", Integer.toHexString(System.identityHashCode((BaseNeatKnot)this.src)), Integer.toHexString(System.identityHashCode((BaseNeatKnot)this.dest)));
    }

    @Override
    public BaseNeatConnection copy() {
        return new BaseNeatConnection(this.inovNum,this.src,this.dest,this.val,this.grad,this.isActive);
    }

    public boolean isActive() {
        return this.isActive;
    }

    @Override
    public List<Connection> insert(Knot b, BiFunction<Knot, Knot, Connection> factory) {
        this.isActive = false;
        return super.insert(b, factory);
    }
}
