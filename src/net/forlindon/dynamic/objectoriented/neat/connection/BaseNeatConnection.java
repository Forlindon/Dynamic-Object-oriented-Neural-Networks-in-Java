package net.forlindon.dynamic.objectoriented.neat.connection;

import net.forlindon.dynamic.objectoriented.neat.genetic.Genotype;
import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neural.networks.connection.BaseConnection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

public class BaseNeatConnection extends BaseConnection implements Genotype {

    private final int inovNum;

    public BaseNeatConnection(InnovationSource innovationSource, Knot src, Knot dest) {
        super(src, dest);
        this.inovNum = innovationSource.getNext();
    }

    protected BaseNeatConnection(int inovNum, Knot src, Knot dest, double val, double grad) {
        super(src, dest);
        this.inovNum = inovNum;
        this.val = val;
        this.grad = grad;
    }

    @Override
    public int getInnovationNumber() {
        return this.inovNum;
    }

    @Override
    public void mutate() {
        this.push((Math.random()*0.1)-0.05);
    }

    @Override
    public String toString() {
        return String.format("%d: %s", this.inovNum, super.toString());
    }

    @Override
    public BaseNeatConnection copy() {
        return new BaseNeatConnection(this.inovNum,this.src,this.dest,this.val,this.grad);
    }
}
