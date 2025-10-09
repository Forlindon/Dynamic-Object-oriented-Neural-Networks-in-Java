package net.forlindon.dynamic.objectoriented.neat.knot;

import net.forlindon.dynamic.objectoriented.neat.TriFunction;
import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.genetic.Genotype;
import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neat.tensor.NeatSimpleTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.BaseKnot;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class BaseNeatKnot extends BaseKnot implements Genotype {

    private final InnovationSource paramSrc;
    private final int inovNum;

    public BaseNeatKnot(InnovationSource paramSrc, int id) {
        super(id);
        this.paramSrc = paramSrc;
        this.inovNum = paramSrc.getNext();
        this.BIAS = getNeatBIAS();
    }

    protected BaseNeatKnot(InnovationSource innovationSource,int inovNum, int id, Tensor activation, Tensor bias) {
        super(id);
        this.paramSrc = innovationSource;
        this.inovNum = inovNum;
        this.OUT = activation;
        this.BIAS = bias;
    }

    @Override
    public Tensor getBIAS() {
        return null;
    }

    public Tensor getNeatBIAS() {
        return new NeatSimpleTensor(this.paramSrc, Math.random()*0.1);
    }

    @Override
    public void add(Connection c) {
        if (c instanceof BaseNeatConnection) super.add(c);
    }

    public void connect(Knot other, TriFunction<InnovationSource, Knot, Knot, Connection> factory) {
        if (other.id() == this.id()) throw new IllegalArgumentException("Invalid LAYER_ID");
        Connection c = factory.apply(this.paramSrc, this, other);
        if (this.OUTBOUND.contains(c)) throw new IllegalArgumentException("No duplicate connections");
        this.OUTBOUND.add(c);
    }

    @Override
    public int getInnovationNumber() {
        return this.inovNum;
    }

    @Override
    public void mutate() {
        this.BIAS.push(Math.random()*0.1-0.05);
    }

    @Override
    public String toString() {
        return String.format("%d: %s", this.inovNum, super.toString());
    }

    public BaseNeatKnot copy() {
        BaseNeatKnot baseNeatKnot = new BaseNeatKnot(this.paramSrc, this.inovNum, this.id(), this.OUT.copy(), this.BIAS.copy());
        baseNeatKnot.OUTBOUND.addAll(this.OUTBOUND.stream().map(x -> (BaseNeatConnection) x).map(BaseNeatConnection::copy).toList());
        return baseNeatKnot;
    }
}
