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

public class SequentialNeatLayer extends BaseNeatLayer {

    private final List<BaseNeatLayer> neatLayers;

    public SequentialNeatLayer(InnovationSource paramSrc) {
        super(paramSrc, -1);
        this.neatLayers = new ArrayList<>();
    }

    protected SequentialNeatLayer(InnovationSource paramSrc, List<BaseNeatLayer> baseNeatLayers) {
        super(paramSrc, -1);
        this.neatLayers = baseNeatLayers.stream().map(BaseNeatLayer::copy).toList();
    }

    @Override
    public void forward() {
        for (Layer l : this.neatLayers) {
            l.forward();
        }
    }

    @Override
    public void forward(double[] inputs) {
        this.clean();
        this.neatLayers.getFirst().forward(inputs);
        for (int i = 1; i < this.neatLayers.size(); i++) {
            this.neatLayers.get(i).forward();
        }
    }

    @Override
    public void backward(boolean init) {
        if (init) this.neatLayers.forEach(l -> l.getKNOTS().forEach(knot -> knot.OUT.pushGrad(1)));
        this.backward();
    }

    @Override
    public void backward() {
        for (int i = this.neatLayers.size()-1; i >= 0; i--) {
            Layer l = this.neatLayers.get(i);
            l.backward(false);
        }
    }

    @Override
    public List<net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot> getKNOTS() {
        List<net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot> ks = new ArrayList<>();
        for (Layer l : this.neatLayers) {
            ks.addAll(l.getKNOTS());
        }
        return ks;
    }

    @Override
    public List<BaseNeatConnection> getGens() {
        List<BaseNeatConnection> genoTypes = new ArrayList<>();
        for (BaseNeatLayer l : this.neatLayers) {
            genoTypes.addAll(l.getGens());
        }
        return genoTypes;
    }

    public void add(BaseNeatLayer l) {
        if (this.id() != l.id()) {
            this.neatLayers.add(l);
            this.KNOTS.addAll(l.getKNOTS());
        }
    }

    public void addLayer(Function<InnovationSource, BaseNeatLayer> factory) {
        this.add(factory.apply(this.PARAM_SRC));
    }

    @Override
    public void add(BiFunction<InnovationSource, Integer, BaseNeatKnot> factory) {
    }

    public void fullyConnect() {
        this.fullyConnect(BaseNeatConnection::new);
    }

    public void fullyConnect(TriFunction<InnovationSource, BaseNeatKnot, BaseNeatKnot, Connection> factory) {
        for (int i = 1; i < this.neatLayers.size(); i++) {
            this.neatLayers.get(i-1).fullConnect(this.neatLayers.get(i),factory);
        }
    }

    @Override
    public SequentialNeatLayer copy() {
        return new SequentialNeatLayer(this.PARAM_SRC,this.neatLayers);
    }

    @Override
    public void add(Knot k) {
        int id = k.id();
        BaseNeatLayer layer = getLayer(id);
        if (this.neatLayers.isEmpty() || !this.neatLayers.contains(layer)) {
            this.neatLayers.add(layer);
        }
        layer.add(k);
    }

    public BaseNeatLayer getLayer(int id) {
        for (BaseNeatLayer baseNeatLayer : this.neatLayers) {
            if (baseNeatLayer.id() == id) return baseNeatLayer;
        }
        return new BaseNeatLayer(this.PARAM_SRC, id, new ArrayList<>());
    }

    public BaseNeatLayer getLastLayer() {
        return this.neatLayers.getLast();
    }

    public BaseNeatLayer getFirstLayer() {
        return this.neatLayers.getFirst();
    }

    @Override
    public void readValues(double[] vals) {
        this.getLastLayer().readValues(vals);
    }

    @Override
    public String toString() {
        return this.neatLayers.toString();
    }

    @Override
    public void clean() {
        this.neatLayers.forEach(BaseLayer::clean);
    }
}
