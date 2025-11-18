package net.forlindon.dynamic.objectoriented.neat.connection;

import net.forlindon.dynamic.objectoriented.neat.genetic.Genotype;
import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neat.genetic.MutationFactory;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neural.networks.connection.BaseConnection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

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
    public int getInnovationNumber() {
        return this.inovNum;
    }

    @Override
    public void mutate() {
        if (Math.random() < 0.9) this.push(MutationFactory.N()*0.1);
        else this.val = MutationFactory.N()*0.5;
        if (Math.random() < 0.25) this.isActive = !this.isActive;
    }

    @Override
    public void ff() {
        if (isActive()) super.ff();
    }

    @Override
    public double getMutationRate() {
        return 0.2;
    }

    @Override
    public String toString() {
        return String.format("w: %g", this.val);
    }

    @Override
    public BaseNeatConnection copy() {
        return new BaseNeatConnection(this.inovNum,this.src,this.dest,this.val,this.grad,this.isActive);
    }

    public boolean isActive() {
        return this.isActive;
    }

}
