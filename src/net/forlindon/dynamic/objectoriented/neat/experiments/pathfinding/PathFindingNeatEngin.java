package net.forlindon.dynamic.objectoriented.neat.experiments.pathfinding;

import net.forlindon.dynamic.objectoriented.neat.NeatEngin;
import net.forlindon.dynamic.objectoriented.neat.genetic.Genome;
import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neat.genetic.SpeciesManager;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.knot.SigNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.knot.TanhNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.visuals.NetWrapper;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.ReluTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.TanhTensor;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public class PathFindingNeatEngin extends NeatEngin {

    Game game;

    public PathFindingNeatEngin(Game game, NetWrapper netWrapper) {
        super(6, 2, 300, null, 1, TanhNeatKnot::new);
        SpeciesManager.setDefaultActivation(TanhTensor::new);
        this.game = game;
        netWrapper.set(this.phenoTypeBuilder.build(this.speciesManager.getFittest()).get());
        this.speciesManager.SPECIATION_BORDER = 2;
        this.speciesManager.c1 = 1.1;
        this.speciesManager.c3 = 0.6;
    }

    @Override
    public Genome run() {
        this.speciesManager.getSpecies().forEach(species -> species.genomes.forEach(genome -> genome.setFitness(0)));
        game.evaluate(this.speciesManager.getSpecies(), this.phenoTypeBuilder);
        do {
        }
        while (game.t.isRunning());
        Genome fittest = this.getSpeciesManager().getFittest().copy();
        this.getSpeciesManager().sort();
        this.getSpeciesManager().removeWeakest();
        this.getSpeciesManager().populate();
        this.getSpeciesManager().mutate();
        if (++gen % report == 0) System.out.printf("Gen: %d, Av: %s, Max: %g, Max Delta: %.5g, Species: %d, Inov: %s, Manager: %s, Deaths: %d\n", gen, speciesManager.averageFitness(), fittest.getFitness(), speciesManager.getMaxDelta(), speciesManager.getNumOfSpecies(), innovationSource, speciesManager, game.players.stream().filter(player -> player.dead).count());
        return fittest;
    }
}
