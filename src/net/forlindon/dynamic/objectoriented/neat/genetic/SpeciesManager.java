package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.SimpleActivationTensor;

import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;

public class SpeciesManager {

    public double SPECIATION_BORDER = 3;
    private static Supplier<Tensor> DEFAULT_ACTIVATION = SimpleActivationTensor::new;

    protected final List<Species> species;

    protected final int MAX_POPULATION;

    protected MutationFactory mutationFactory = new MutationFactory();

    public double c1 = 1; // disjoint Genes
    public double c2 = 1; // excess Genes
    public double c3 = 0.4; // weights
    // public static double c4 = 0; // nodes

    public SpeciesManager(int maxPopulation) {
        this.MAX_POPULATION = maxPopulation;
        this.species = new ArrayList<>();
    }

    @Override
    public String toString() {
        return this.species.stream().map(species1 -> species1.genomes.size()).toList().toString();
    }

    public void populate() {
        double[] avFit = this.species.stream().mapToDouble(x -> x.getFitness() / x.genomes.size()).toArray();
        double sum = Arrays.stream(avFit).sum();

        int[] offspringCount = new int[species.size()];
        for (int i = 0; i < species.size(); i++) {
            offspringCount[i] = Math.max(1, (int)(avFit[i] / sum * MAX_POPULATION));
        }

        while (getPopulationSize() < MAX_POPULATION) {
            for (int i = 0; i < species.size(); i++) {
                Species s = species.get(i);
                if (offspringCount[i] <= 0) continue;
                addToSpecies(s);
                offspringCount[i]--;
                if (getPopulationSize() >= MAX_POPULATION) break;
            }
        }
    }

    private void addToSpecies(Species s) {
        if (Math.random() < 0.75 && s.genomes.size() >= 2) {
            s.genomes.add(Genome.crossOver(s.getFittest(), s.getSecondFittest()));
        } else {
            Genome g = s.sample(0.3).copy();
            this.mutationFactory.mutateSelf(g);
            s.genomes.add(g);
        }
    }

    public int getPopulationSize() {
        return this.species.stream().mapToInt(x -> x.genomes.size()).sum();
    }

    public void sort() {
        List<Genome> genomes = this.species.stream().map(species1 -> species1.genomes).flatMap(List::stream).toList();
        this.species.forEach(species1 -> species1.genomes.clear());
        for (Genome g : genomes) {
            boolean assigned = false;
            for (Species s : species) {
                if (s.contains(g)) {
                    s.add(g);
                    assigned = true;
                    break;
                }
            }
            if (assigned) continue;
            Species next = new Species(this, g);
            next.add(g);
            this.species.add(next);
        }
        this.species.removeIf(species1 -> species1.genomes.isEmpty());
        this.species.forEach(s -> {
            Genome min = s.genomes.getFirst();
            for (Genome g : s.genomes) {
                if (s.reference.calcDelta(min, this.c1, this.c2, this.c3) < s.reference.calcDelta(g, this.c1, this.c2, this.c3)) {
                    min = g;
                }
            }
            s.reference = min;
        });
    }

    public void evaluate(Function<Genome, Double> fitnessSupplier) {
        this.species.forEach(s -> s.genomes.forEach(y -> y.fitness = fitnessSupplier.apply(y)));
    }

    public void removeWeakest() {
        this.species.removeIf(species1 -> species1.genomes.isEmpty());
        this.species.forEach(Species::removeWeakest);
    }

    public static void probability(double[] array) {
        normalize(array);
        softmax(array);
    }

    public static void normalize(double[] array) {
        double min = Arrays.stream(array).min().getAsDouble();
        for (int i = 0; i < array.length; i++) {
            array[i]-=min;
        }
        double sum = Arrays.stream(array).sum();
        if (sum == 0) sum++;
        for (int i = 0; i < array.length; i++) {
            array[i]/=sum;
        }
    }

    public static void softmax(double[] array) {
        for (int i = 0; i < array.length; i++) {
            array[i]=Math.exp(array[i]);
        }
        double sum = Arrays.stream(array).sum();
        for (int i = 0; i < array.length; i++) {
            array[i]/=sum;
        }
    }

    public void init(Species init) {
        this.species.add(init);
        Genome g = init.genomes.getFirst();
        g.fitness = 1e-5;
        while (getPopulationSize() < MAX_POPULATION) {
            Genome x = g.copy();
            x.copyValues();
            this.species.getFirst().add(x);
        }
    }

    public void init() {
        this.init(this.species.getFirst());
    }

    public void mutate() {
        this.species.forEach(sp -> sp.genomes.forEach(mutationFactory::mutateSelf));
    }

    public double averageFitness() {
        return this.species.stream().mapToDouble(Species::averageFitness).sum()/this.species.size();
    }

    public Genome getFittest() {
        List<Genome> genomes = new ArrayList<>(this.species.stream().mapToInt(value -> value.genomes.size()).sum());
        for (Species sp : this.species) {
            genomes.addAll(sp.genomes);
        }
        genomes.sort(new GenomeFitnessComparator());
        return genomes.getLast();
    }

    public int getNumOfSpecies() {
        return this.species.size();
    }

    public double getMaxDelta() {
        return this.species.stream().mapToDouble(Species::calcMaxDelta).max().getAsDouble();
    }

    public List<Species> getSpecies() {
        return new ArrayList<>(this.species);
    }

    public double getAverageFitness() {
        return this.species.stream().mapToDouble(Species::averageFitness).sum()/this.species.size();
    }

    public static void setDefaultActivation(Supplier<Tensor> supplier) {
        DEFAULT_ACTIVATION = supplier;
    }

    public static Tensor getDefaultActivation() {
        return DEFAULT_ACTIVATION.get();
    }

    public List<Genome> genomes() {
        return this.species.stream().map(species1 -> species1.genomes).flatMap(List::stream).toList();
    }

    public MutationFactory getMutationFactory() {
        return this.mutationFactory;
    }
}
