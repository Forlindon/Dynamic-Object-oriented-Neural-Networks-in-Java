package net.forlindon.dynamic.objectoriented.neat;

import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.genetic.*;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.layer.NeatLinearLayer;
import net.forlindon.dynamic.objectoriented.neat.layer.SequentialNeatLayer;
import net.forlindon.dynamic.objectoriented.neat.visuals.NetWrapper;

import java.util.function.BiFunction;
import java.util.function.Function;

public class NeatEngin {

    protected InnovationSource innovationSource = new InnovationSource();

    protected SpeciesManager speciesManager;

    protected PhenoTypeBuilder phenoTypeBuilder = new PhenoTypeBuilder();

    protected Function<Genome, Double> fittnessSupplier;

    protected int gen = 0;
    protected int report;

    public NeatEngin(int in, int out, int populationSize, Function<Genome, Double> fitnessSupplier, int report, BiFunction<InnovationSource,Integer,BaseNeatKnot> factory) {
        this.report = report;
        this.fittnessSupplier = fitnessSupplier;
        this.speciesManager = new SpeciesManager(populationSize);

        SequentialNeatLayer net = new SequentialNeatLayer(innovationSource);
        net.addLayer(src -> new NeatLinearLayer(in, src, BaseNeatKnot::new));
        net.addLayer(src -> new NeatLinearLayer(out, src, factory));
        net.fullyConnect(BaseNeatConnection::new);

        Genome g = new Genome(net);
        Species species = new Species(speciesManager, g);
        species.add(g);

        speciesManager.init(species);

        this.speciesManager.mutate();
    }

    public Genome run() {
        this.speciesManager.evaluate(this.fittnessSupplier);
        Genome fittest = this.speciesManager.getFittest().copy();
        this.speciesManager.sort();
        this.speciesManager.removeWeakest();
        this.speciesManager.populate();
        this.speciesManager.mutate();
        if (++gen % report == 0) System.out.printf("Gen: %d, Av: %s, Max Delta: %.5g, Species: %d, Inov: %s, Manager: %s\n", gen, speciesManager.averageFitness(), speciesManager.getMaxDelta(), speciesManager.getNumOfSpecies(), innovationSource, speciesManager);
        return fittest;
    }

    public synchronized void update(NetWrapper netWrapper) {
        Genome fittest = this.speciesManager.getFittest().copy();
        PhenoType phenoType = this.phenoTypeBuilder.build(fittest);
        netWrapper.set(phenoType.get());
    }

    public PhenoTypeBuilder getPhenoTypeBuilder() {
        return phenoTypeBuilder;
    }

    public InnovationSource getInnovationSource() {
        return innovationSource;
    }

    public int getReport() {
        return report;
    }

    public int getGen() {
        return gen;
    }

    public Function<Genome, Double> getFittnessSupplier() {
        return fittnessSupplier;
    }

    public SpeciesManager getSpeciesManager() {
        return speciesManager;
    }
}
