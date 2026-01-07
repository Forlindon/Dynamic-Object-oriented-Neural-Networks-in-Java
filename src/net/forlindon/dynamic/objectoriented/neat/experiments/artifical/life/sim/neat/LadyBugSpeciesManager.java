package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.neat;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects.LadyBug;
import net.forlindon.dynamic.objectoriented.neat.genetic.*;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.knot.ReluNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.knot.SigNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.knot.TanhNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.layer.BaseNeatLayer;
import net.forlindon.dynamic.objectoriented.neat.layer.NeatLinearLayer;
import net.forlindon.dynamic.objectoriented.neat.layer.SequentialNeatLayer;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.ReluTensor;

import java.util.function.Function;

public class LadyBugSpeciesManager extends SpeciesManager {

    public PhenoTypeBuilder phenoTypeBuilder = new PhenoTypeBuilder();

    public LadyBugSpeciesManager() {
        super(200);
        InnovationSource src = new InnovationSource();
        SequentialNeatLayer sequentialNeatLayer = new SequentialNeatLayer(src);
        SpeciesManager.setDefaultActivation(ReluTensor::new);
        sequentialNeatLayer.addLayer(innovationSource -> new NeatLinearLayer(LadyBug.OBS_SPACE, innovationSource, BaseNeatKnot::new));
        sequentialNeatLayer.addLayer(innovationSource -> new NeatLinearLayer(32, innovationSource, SigNeatKnot::new));
        BaseNeatLayer out = new BaseNeatLayer(src, src.getNext());
        out.add(TanhNeatKnot::new);
        out.add(TanhNeatKnot::new);
        out.add(SigNeatKnot::new);
        out.add(SigNeatKnot::new);
        sequentialNeatLayer.add(out);
        sequentialNeatLayer.fullyConnect();
        Genome g = new Genome(sequentialNeatLayer);
        Species species = new Species(this, g);
        species.add(g);
        this.mutationFactory.setActivationAdaptation(true);
        this.species.add(species);
        this.SPECIATION_BORDER = 1;
        this.c1 = 0.3;
        this.c2 = 0.3;
        this.c3 = 1;
    }

    @Override
    public void evaluate(Function<Genome, Double> fitnessSupplier) {
    }

    @Override
    public void removeWeakest() {
    }

    @Override
    public void populate() {
    }

    @Override
    public void init() {
    }

    @Override
    public void init(Species init) {
    }

    @Override
    public void mutate() {
    }

    public void getGenome(LadyBug ladyBug) {
        Species first = this.species.getFirst();
        Genome g = first.reference.copy();
        this.mutationFactory.mutateSelf(g);
        first.add(g);
        ladyBug.genome = g;
        ladyBug.phenoType = this.phenoTypeBuilder.build(g);
    }

    public void add(Genome g) {
        this.species.getFirst().add(g);
        this.sort();
    }

    public void remove(LadyBug ladyBug) {
        for (Species species1 : this.species) {
            species1.genomes.removeIf(genome -> genome == ladyBug.genome);
        }
        sort();
    }

    @Override
    public String toString() {
        return String.format("PopulationSize: %d, Species: %d, Max Delta: %g", getPopulationSize(), getNumOfSpecies(), getMaxDelta());
    }
}
