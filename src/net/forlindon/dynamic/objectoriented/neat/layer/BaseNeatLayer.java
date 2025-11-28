package net.forlindon.dynamic.objectoriented.neat.layer;

import net.forlindon.dynamic.objectoriented.neat.TriFunction;
import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;
import net.forlindon.dynamic.objectoriented.neural.networks.layer.BaseLayer;
import net.forlindon.dynamic.objectoriented.neural.networks.layer.Layer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public class BaseNeatLayer extends BaseLayer {

    public final InnovationSource PARAM_SRC;

    public BaseNeatLayer(InnovationSource paramSrc, int id) {
        super(id);
        this.PARAM_SRC = paramSrc;
    }

    protected BaseNeatLayer(InnovationSource paramSrc, int id, List<Knot> list) {
        super(id);
        this.PARAM_SRC = paramSrc;
        this.KNOTS.addAll(list.stream().map(x -> (BaseNeatKnot) x).map(BaseNeatKnot::copy).toList());
    }

    @Override
    public void add(Function<Integer, net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot> factory) {
        throw new RuntimeException("Not valid method for neat");
    }

    public void add(BiFunction<InnovationSource,Integer, BaseNeatKnot> factory) {
        this.KNOTS.add(factory.apply(this.PARAM_SRC, this.id()));
    }

    @Override
    public void add(Knot k) {
        if (k instanceof BaseNeatKnot) super.add(k);
    }

    public List<BaseNeatConnection> getGens() {
        List<BaseNeatConnection> gens = new ArrayList<>();
        for (Knot k : this.KNOTS) {
            gens.addAll(k.getConnections().stream().map(x -> (BaseNeatConnection) x).toList());
        }
        return gens;
    }

    public void fullConnect(Layer other, TriFunction<InnovationSource, BaseNeatKnot, BaseNeatKnot, Connection> factory) {
        for (net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot a : this.KNOTS) {
            BaseNeatKnot c = (BaseNeatKnot) a;
            for (Knot b : other.getKNOTS()) {
                c.connect((BaseNeatKnot) b, factory);
            }
        }
    }

    @Override
    public List<Knot> getKNOTS() {
        return super.getKNOTS();
    }

    public BaseNeatLayer copy() {
        return new BaseNeatLayer(this.PARAM_SRC,this.id(),getKNOTS());
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof BaseNeatLayer baseNeatLayer && baseNeatLayer.id() == this.id();
    }
}
