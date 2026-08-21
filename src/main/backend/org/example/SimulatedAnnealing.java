package org.example;

import org.example.Plant;

import java.util.*;

public class SimulatedAnnealing {

    private double T0;       // initial temperature
    private double alpha;    // cooling rate
    private int maxIter;     // max iterations

    public SimulatedAnnealing(double T0, double alpha, int maxIter) {
        this.T0 = T0;
        this.alpha = alpha;
        this.maxIter = maxIter;
    }

    // cost = plants_missed + normalized_distance + extra_watered
    //   plants_missed  = "needs water" plants NOT in the sequence
    //   extra_watered  = plants in sequence predicted "no water"
    //   distance       = sum of Euclidean distances, normalized by canvas diagonal
    private double cost(int[] order, Plant[] plants, boolean[] needsWater, double maxDist) {
        Set<Integer> inSeq = new HashSet<>();
        for (int id : order) inSeq.add(id);

        int missed = 0;
        for (int i = 0; i < plants.length; i++)
            if (needsWater[i] && !inSeq.contains(i)) missed++;

        int extra = 0;
        for (int id : order)
            if (!needsWater[id]) extra++;

        double dist = 0;
        for (int i = 0; i < order.length - 1; i++)
            dist += euclidean(plants[order[i]], plants[order[i + 1]]);

        double normDist = dist / maxDist;
        return missed + normDist + extra;
    }

    private double euclidean(Plant a, Plant b) {
        return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
    }

    // generate neighbour: either swap order of two plants, or replace one with an outside plant
    private int[] generateNeighbour(int[] order, int totalPlants) {
        int[] neighbour = order.clone();
        Set<Integer> inSeq = new HashSet<>();
        for (int id : neighbour) inSeq.add(id);

        List<Integer> outside = new ArrayList<>();
        for (int i = 0; i < totalPlants; i++)
            if (!inSeq.contains(i)) outside.add(i);

        Random rand = new Random();
        if (outside.isEmpty() || rand.nextDouble() < 0.5) {
            // reorder: swap two positions
            if (neighbour.length < 2) return neighbour;
            int i = rand.nextInt(neighbour.length);
            int j;
            do { j = rand.nextInt(neighbour.length); } while (j == i);
            int tmp = neighbour[i]; neighbour[i] = neighbour[j]; neighbour[j] = tmp;
        } else {
            // replace: swap one inside plant with one outside
            int i = rand.nextInt(neighbour.length);
            int j = rand.nextInt(outside.size());
            neighbour[i] = outside.get(j);
        }
        return neighbour;
    }

    // result wrapper so we can return both the order and cost history
    public static class Result {
        public int[]    order;
        public double[] costHistory;
        public Result(int[] order, double[] costHistory) {
            this.order       = order;
            this.costHistory = costHistory;
        }
    }

    public Result optimize(Plant[] plants, boolean[] needsWater, int numToWater, double maxDist) {
        Random rand = new Random();

        // start with plants that need water first, fill up to numToWater
        List<Integer> needIds = new ArrayList<>();
        List<Integer> otherIds = new ArrayList<>();
        for (int i = 0; i < plants.length; i++) {
            if (needsWater[i]) needIds.add(i);
            else otherIds.add(i);
        }
        Collections.shuffle(needIds, rand);
        Collections.shuffle(otherIds, rand);
        needIds.addAll(otherIds);

        int[] current = new int[numToWater];
        for (int i = 0; i < numToWater; i++) current[i] = needIds.get(i);

        double curCost  = cost(current, plants, needsWater, maxDist);
        int[]  best     = current.clone();
        double bestCost = curCost;
        double T        = T0;

        double[] costHistory = new double[maxIter + 1];
        costHistory[0] = curCost;

        for (int iter = 0; iter < maxIter; iter++) {
            int[]  neighbour = generateNeighbour(current, plants.length);
            double newCost   = cost(neighbour, plants, needsWater, maxDist);
            double delta     = newCost - curCost;

            boolean accept = (delta < 0) || (rand.nextDouble() < Math.exp(-delta / T));
            if (accept) {
                current = neighbour;
                curCost = newCost;
                if (curCost < bestCost) {
                    best     = current.clone();
                    bestCost = curCost;
                }
            }
            costHistory[iter + 1] = curCost;
            T *= alpha;
        }

        System.out.println("Best cost: " + bestCost);
        return new Result(best, costHistory);
    }
}