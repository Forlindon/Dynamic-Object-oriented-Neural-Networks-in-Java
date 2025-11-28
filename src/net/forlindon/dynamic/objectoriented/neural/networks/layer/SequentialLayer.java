package net.forlindon.dynamic.objectoriented.neural.networks.layer;

import net.forlindon.dynamic.objectoriented.neat.layer.BaseNeatLayer;
import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public class SequentialLayer extends Layer {

    protected final List<Layer> LAYERS;

    @SafeVarargs
    public SequentialLayer(Function<Integer, Layer>... factories) {
        super(-1);
        this.LAYERS = new ArrayList<>();
        for (int i = 0; i < factories.length; i++) {
            this.LAYERS.add(factories[i].apply(i));
            this.KNOTS.addAll(LAYERS.get(i).KNOTS);
        }
    }

    protected SequentialLayer() {
        super(-1);
        this.LAYERS = new ArrayList<>();
    }

    public SequentialLayer(Layer... l) {
        super(-1);
        this.LAYERS = new ArrayList<>();
        this.LAYERS.addAll(Arrays.asList(l));
    }

    public void fullConnect(BiFunction<Knot, Knot, Connection> factory) {
        for (int i = 0; i < this.LAYERS.size(); i++) {
            this.LAYERS.get(i).fullConnect(this.LAYERS.get(i+1), factory);
        }
    }

    @Override
    public void fullConnect(Layer other, BiFunction<Knot, Knot, Connection> factory) {
        this.LAYERS.getLast().fullConnect(other,factory);
    }

    @Override
    public void forward() {
        for (Layer l : this.LAYERS) {
            l.forward();
        }
    }

    @Override
    public void forward(double[] inputs) {
        this.LAYERS.getFirst().forward(inputs);
        for (int i = 1; i < this.LAYERS.size(); i++) {
            this.LAYERS.get(i).forward();
        }
    }

    @Override
    public void backward(boolean init) {
        if (init) this.LAYERS.forEach(l -> l.KNOTS.forEach(knot -> knot.OUT.pushGrad(1)));
        this.backward();
    }

    @Override
    public void backward() {
        for (int i = this.LAYERS.size()-1; i >= 0; i--) {
            Layer l = this.LAYERS.get(i);
            l.backward(false);
        }
    }

    @Override
    public void readValues(double[] vals) {
        this.LAYERS.getLast().readValues(vals);
    }

    public Layer get(int idx) {
        return this.LAYERS.get(idx);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder("{\n");
        for (Layer l : this.LAYERS) {
            stringBuilder.append(l);
            stringBuilder.append('\n');
        }
        stringBuilder.append("}");
        return stringBuilder.toString();
    }

    @Override
    public List<Knot> getKNOTS() {
        List<Knot> knots = new ArrayList<>();
        for (Layer l : this.LAYERS) {
            knots.addAll(l.getKNOTS());
        }
        return knots;
    }

    @Override
    public List<Tensor> getParameters() {
        List<Tensor> ts = new ArrayList<>();
        for (Layer l : this.LAYERS) {
            ts.addAll(l.getParameters());
        }
        return ts;
    }

    @Override
    public void clean() {
        for (Layer l : this.LAYERS) {
            l.clean();
        }
    }

    @Override
    public void add(Knot k) {
        int id = k.id();
        Layer layer = getLayer(id);
        if (this.LAYERS.isEmpty() || !this.LAYERS.contains(layer)) {
            this.LAYERS.add(layer);
        }
        layer.add(k);
    }

    public Layer getLayer(int id) {
        for (Layer baseNeatLayer : this.LAYERS) {
            if (baseNeatLayer.id() == id) return baseNeatLayer;
        }
        return new BaseLayer(id);
    }

    @Override
    public Layer copy() {
        SequentialLayer sequentialLayer = new SequentialLayer();
        List<Knot> prev = getKNOTS();
        for (Knot k : prev) {
            sequentialLayer.add(k.copy());
        }
        for (Knot k : sequentialLayer.KNOTS) {
            for (Connection c : k.getConnections()) {
                c.setDest(sequentialLayer.KNOTS.get(prev.indexOf(c.dest())));
            }
        }
        return sequentialLayer;
    }

    public Layer getLast() {
        return this.LAYERS.getLast();
    }

    public void add(int idx, Function<Integer,Knot> factory) {
        this.LAYERS.get(idx).add(factory);
    }


    private void addLayer(Layer l) {
        this.LAYERS.add(l);
    }

}
