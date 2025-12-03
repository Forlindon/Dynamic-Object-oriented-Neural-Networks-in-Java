package net.forlindon.dynamic.objectoriented.neat.knot;

import net.forlindon.dynamic.objectoriented.neat.TriFunction;
import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.genetic.Genotype;
import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neat.genetic.MutationFactory;
import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.BaseKnot;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.*;

public class BaseNeatKnot extends BaseKnot implements Genotype {

    protected final InnovationSource paramSrc;
    protected final int inovNum;

    public BaseNeatKnot(InnovationSource paramSrc, int id) {
        super(id);
        this.paramSrc = paramSrc;
        this.inovNum = paramSrc.getNext();
    }

    protected BaseNeatKnot(InnovationSource innovationSource, int inovNum, int id, Tensor activation, Tensor bias) {
        super(id);
        this.paramSrc = innovationSource;
        this.inovNum = inovNum;
        this.OUT = activation;
        this.BIAS = bias;
    }

    @Override
    public void add(Connection c) {
        if (c instanceof BaseNeatConnection) super.add(c);
    }

    public void connect(BaseNeatKnot other, TriFunction<InnovationSource, BaseNeatKnot, BaseNeatKnot, Connection> factory) {
        if (other.id() == this.id()) throw new IllegalArgumentException("Invalid LAYER_ID");
        Connection c = factory.apply(this.paramSrc, this, other);
        if (this.OUTBOUND.contains(c)) throw new IllegalArgumentException("No duplicate connections");
        this.OUTBOUND.add(c);
    }

    @Override
    public int inov() {
        return this.inovNum;
    }

    @Override
    public void mutate() {
        this.BIAS.push(0.1*MutationFactory.N());
    }

    @Override
    public double getMutationRate() {
        return 0.9;
    }

    public static double getMutationActivationRate() {
        return 0.2;
    }

    @Override
    public String toString() {
        // return ""+this.inovNum;
        // return String.format("%d: %s", this.inovNum, super.toString());
        return String.format("%d: %s", this.hashCode(), super.toString());
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof BaseNeatKnot o && this.inovNum == o.inovNum;
    }

    @Override
    public int hashCode() {
        return this.inovNum;
    }

    public BaseNeatKnot copy() {
        BaseNeatKnot baseNeatKnot = new BaseNeatKnot(this.paramSrc, this.inovNum, this.id(), this.OUT.copy(), this.BIAS.copy());
        baseNeatKnot.OUTBOUND.addAll(this.OUTBOUND.stream().map(x -> (BaseNeatConnection) x.copy()).toList());
        baseNeatKnot.OUTBOUND.forEach(x -> x.setSrc(baseNeatKnot));
        return baseNeatKnot;
    }

    public static void mutateActivation(BaseNeatKnot baseNeatKnot) {
        double r = Math.random();
        if (r < 0.1) baseNeatKnot.OUT = new ReluTensor();
        if (r < 0.2) baseNeatKnot.OUT = new SigTensor();
        if (r < 0.3) baseNeatKnot.OUT = new TanhTensor();
        if (r < 0.4) baseNeatKnot.OUT = new ParametricReluTensor(Math.random()-0.5);
        if (r < 0.5) baseNeatKnot.OUT = new SimpleActivationTensor();
    }
}
