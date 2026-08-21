package org.example;

import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

import java.io.*;
import java.net.InetSocketAddress;
import java.util.*;

public class Main {

    static int[][] DATASET = {
            {21,8,2,1},{26,13,0,0},{57,25,1,0},{23,47,0,1},{14,42,0,1},
            {35,34,2,1},{54,5,0,0},{13,6,0,1},{25,1,2,1},{73,30,1,0},
            {50,5,0,0},{14,23,2,1},{67,12,2,0},{64,40,2,1},{18,34,1,1},
            {45,39,1,1},{46,19,0,0},{14,15,0,1},{39,37,2,1},{64,30,1,0},
            {13,37,0,1},{31,27,1,1},{26,34,1,1},{26,18,2,1},{23,24,1,1},
            {43,31,2,1},{43,29,1,0},{63,45,1,1},{41,13,2,0},{25,25,1,1},
            {75,38,1,1},{26,35,1,1},{76,10,2,0},{74,42,1,1},{15,44,1,1},
            {24,19,0,0},{38,40,2,1},{49,1,1,0},{61,8,1,0},{23,36,0,0},
            {79,48,1,1},{63,48,2,1},{72,34,0,0},{43,48,1,1},{55,15,2,0},
            {69,36,1,0},{49,22,1,0},{35,12,1,1},{23,37,2,1},{57,24,2,0},
            {40,41,2,1},{20,16,0,0},{25,17,0,0},{51,39,0,0},{28,21,1,1},
            {45,0,0,0},{38,43,2,1},{22,4,0,0},{36,16,1,1},{59,39,0,0},
            {17,16,0,1},{79,38,0,0},{12,10,2,1},{54,44,2,1},{43,11,1,0},
            {79,3,0,0},{77,10,2,0},{57,4,2,0},{42,6,1,0},{27,41,0,1},
            {27,16,1,1},{21,40,1,1},{15,44,2,1},{30,1,2,1},{15,32,1,1},
            {55,26,2,0},{37,20,2,0},{42,42,1,1},{18,16,1,1},{35,35,0,0},
            {27,5,2,1},{10,48,0,1},{24,33,1,1},{79,42,0,1},{57,27,1,0},
            {27,48,0,1},{43,4,0,0},{48,3,0,0},{61,32,2,1},{61,34,2,1},
            {28,21,1,1},{20,42,1,1},{24,9,0,0},{62,27,0,0},{36,42,2,1},
            {42,38,1,1},{70,17,2,0},{10,21,1,1},{38,12,2,0},{33,28,2,1}
    };

    static Perceptron perceptron = null;

    public static void main(String[] args) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);
        server.createContext("/train",      new TrainHandler());
        server.createContext("/train-stream", new TrainStreamHandler());
        server.createContext("/predict",    new PredictHandler());
        server.createContext("/sa",         new SAHandler());
        server.createContext("/sa-stream",  new SAStreamHandler());
        server.setExecutor(java.util.concurrent.Executors.newCachedThreadPool());
        server.start();
        System.out.println("Server running at http://localhost:8080");
        System.out.println("Open frontend/index.html in your browser");
    }

    // helper: send a single SSE event line
    static void sendEvent(OutputStream out, String data) throws IOException {
        String msg = "data: " + data + "\n\n";
        out.write(msg.getBytes());
        out.flush();
    }

    // ── /train-stream  (GET, SSE) ──────────────────────────────────────────
    // streams one event per epoch: {"epoch":1,"errors":5,"acc":0.81,"w1":...}
    // final event has "done":true and includes testAcc
    static class TrainStreamHandler implements HttpHandler {
        public void handle(HttpExchange exchange) throws IOException {
            Map<String, String> params = parseQuery(exchange.getRequestURI().getQuery());
            double alpha    = Double.parseDouble(params.getOrDefault("alpha",  "0.1"));
            int    maxEpochs= Integer.parseInt(params.getOrDefault("epochs", "100"));
            double split    = Double.parseDouble(params.getOrDefault("split",  "0.8"));

            exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
            exchange.getResponseHeaders().add("Content-Type",  "text/event-stream");
            exchange.getResponseHeaders().add("Cache-Control", "no-cache");
            exchange.sendResponseHeaders(200, 0);
            OutputStream out = exchange.getResponseBody();

            int trainN = (int)(DATASET.length * split);
            double[][] trainX = new double[trainN][3];
            int[]      trainY = new int[trainN];
            double[][] testX  = new double[DATASET.length - trainN][3];
            int[]      testY  = new int[DATASET.length - trainN];

            for (int i = 0; i < DATASET.length; i++) {
                double[] x = Perceptron.normalize(DATASET[i][0], DATASET[i][1], DATASET[i][2]);
                if (i < trainN) { trainX[i] = x; trainY[i] = DATASET[i][3]; }
                else            { testX[i-trainN] = x; testY[i-trainN] = DATASET[i][3]; }
            }

            perceptron = new Perceptron(alpha);
            double[] w = perceptron.getWeights();

            // send initial weights
            sendEvent(out, String.format(
                    "{\"epoch\":0,\"errors\":0,\"acc\":0,\"w1\":%.4f,\"w2\":%.4f,\"w3\":%.4f,\"theta\":%.4f}",
                    w[0], w[1], w[2], perceptron.getTheta()
            ));

            for (int e = 0; e < maxEpochs; e++) {
                int errors = 0;
                for (int i = 0; i < trainX.length; i++) {
                    int pred  = perceptron.predict(trainX[i]);
                    int delta = trainY[i] - pred;
                    if (delta != 0) {
                        perceptron.update(trainX[i], trainY[i]);
                        errors++;
                    }
                }
                double acc = perceptron.accuracy(trainX, trainY);
                w = perceptron.getWeights();

                sendEvent(out, String.format(
                        "{\"epoch\":%d,\"errors\":%d,\"acc\":%.4f,\"w1\":%.4f,\"w2\":%.4f,\"w3\":%.4f,\"theta\":%.4f}",
                        e + 1, errors, acc, w[0], w[1], w[2], perceptron.getTheta()
                ));

                if (errors == 0) break;
            }

            double testAcc = perceptron.accuracy(testX, testY);
            w = perceptron.getWeights();
            sendEvent(out, String.format(
                    "{\"done\":true,\"testAcc\":%.4f,\"w1\":%.4f,\"w2\":%.4f,\"w3\":%.4f,\"theta\":%.4f}",
                    testAcc, w[0], w[1], w[2], perceptron.getTheta()
            ));

            out.close();
        }
    }

    // ── /sa-stream  (POST, SSE) ────────────────────────────────────────────
    // streams one event per iteration (every 10th after iter 30):
    // {"iter":50,"cost":2.4,"order":[0,1,2],"type":"improved"}
    // final event has "done":true
    static class SAStreamHandler implements HttpHandler {
        public void handle(HttpExchange exchange) throws IOException {
            exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
            exchange.getResponseHeaders().add("Content-Type",  "text/event-stream");
            exchange.getResponseHeaders().add("Cache-Control", "no-cache");
            exchange.sendResponseHeaders(200, 0);
            OutputStream out = exchange.getResponseBody();

            if (perceptron == null) {
                sendEvent(out, "{\"error\":\"Train the perceptron first\"}");
                out.close(); return;
            }

            String body = new String(exchange.getRequestBody().readAllBytes());
            Map<String, String> top = parseSimpleJson(body);
            int    numToWater = Integer.parseInt(top.getOrDefault("numToWater", "3"));
            double T0         = Double.parseDouble(top.getOrDefault("T0",       "200"));
            double saAlpha    = Double.parseDouble(top.getOrDefault("alpha",    "0.97"));
            int    maxIter    = Integer.parseInt(top.getOrDefault("maxIter",    "1000"));
            double canvasW    = Double.parseDouble(top.getOrDefault("canvasW",  "560"));
            double canvasH    = Double.parseDouble(top.getOrDefault("canvasH",  "380"));

            List<Plant> plantList  = parsePlants(body);
            Plant[]     plants     = plantList.toArray(new Plant[0]);
            boolean[]   needsWater = new boolean[plants.length];
            int[]       predArr    = new int[plants.length];

            for (int i = 0; i < plants.length; i++) {
                double[] x = Perceptron.normalize(plants[i].moisture, plants[i].lastWatered, plants[i].plantType);
                needsWater[i] = perceptron.predict(x) == 1;
                predArr[i]    = needsWater[i] ? 1 : 0;
            }

            double maxDist = Math.sqrt(canvasW * canvasW + canvasH * canvasH);

            // run SA manually here so we can stream each step
            Random rand = new Random();
            List<Integer> needIds  = new ArrayList<>();
            List<Integer> otherIds = new ArrayList<>();
            for (int i = 0; i < plants.length; i++) {
                if (needsWater[i]) needIds.add(i); else otherIds.add(i);
            }
            Collections.shuffle(needIds,  rand);
            Collections.shuffle(otherIds, rand);
            needIds.addAll(otherIds);

            int numW = Math.min(numToWater, plants.length);
            int[] current = new int[numW];
            for (int i = 0; i < numW; i++) current[i] = needIds.get(i);

            double curCost  = saCost(current, plants, needsWater, maxDist);
            int[]  best     = current.clone();
            double bestCost = curCost;
            double T        = T0;

            sendEvent(out, String.format(
                    "{\"iter\":0,\"cost\":%.4f,\"order\":%s,\"type\":\"start\"}",
                    curCost, arrToJson(current)
            ));

            for (int iter = 0; iter < maxIter; iter++) {
                int[]  neighbour = saNeighbour(current, plants.length, rand);
                double newCost   = saCost(neighbour, plants, needsWater, maxDist);
                double delta     = newCost - curCost;

                String type;
                boolean accept;
                if (delta < 0) {
                    accept = true; type = "improved";
                } else {
                    accept = rand.nextDouble() < Math.exp(-delta / T);
                    type   = accept ? "prob-accept" : "rejected";
                }

                if (accept) {
                    current = neighbour; curCost = newCost;
                    if (curCost < bestCost) { best = current.clone(); bestCost = curCost; }
                }

                T *= saAlpha;

                // stream every step for first 30, then every 50
                if (iter < 30 || iter % 100 == 0) {
                    sendEvent(out, String.format(
                            "{\"iter\":%d,\"cost\":%.4f,\"order\":%s,\"type\":\"%s\",\"T\":%.3f}",
                            iter + 1, curCost, arrToJson(current), type, T
                    ));
                }
            }

            sendEvent(out, String.format(
                    "{\"done\":true,\"best\":%s,\"bestCost\":%.4f,\"predictions\":%s}",
                    arrToJson(best), bestCost, intArrToJson(predArr)
            ));

            out.close();
        }
    }

    // SA cost and neighbour helpers (duplicated here for streaming)
    static double saCost(int[] order, Plant[] plants, boolean[] needsWater, double maxDist) {
        Set<Integer> inSeq = new HashSet<>();
        for (int id : order) inSeq.add(id);
        int missed = 0, extra = 0;
        for (int i = 0; i < plants.length; i++) {
            if (needsWater[i] && !inSeq.contains(i)) missed++;
            if (!needsWater[i] && inSeq.contains(i)) extra++;
        }
        double dist = 0;
        for (int i = 0; i < order.length - 1; i++) {
            double dx = plants[order[i]].x - plants[order[i+1]].x;
            double dy = plants[order[i]].y - plants[order[i+1]].y;
            dist += Math.sqrt(dx*dx + dy*dy);
        }
        return missed + dist / maxDist + extra;
    }

    static int[] saNeighbour(int[] order, int totalPlants, Random rand) {
        int[] n = order.clone();
        Set<Integer> inSeq = new HashSet<>();
        for (int id : n) inSeq.add(id);
        List<Integer> outside = new ArrayList<>();
        for (int i = 0; i < totalPlants; i++) if (!inSeq.contains(i)) outside.add(i);

        if (outside.isEmpty() || rand.nextDouble() < 0.5) {
            if (n.length < 2) return n;
            int i = rand.nextInt(n.length), j;
            do { j = rand.nextInt(n.length); } while (j == i);
            int tmp = n[i]; n[i] = n[j]; n[j] = tmp;
        } else {
            n[rand.nextInt(n.length)] = outside.get(rand.nextInt(outside.size()));
        }
        return n;
    }

    static String arrToJson(int[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) { if (i>0) sb.append(","); sb.append(arr[i]); }
        return sb.append("]").toString();
    }

    static String intArrToJson(int[] arr) { return arrToJson(arr); }

    // ── /train  (POST, kept for compatibility) ─────────────────────────────
    static class TrainHandler implements HttpHandler {
        public void handle(HttpExchange exchange) throws IOException {
            if (!exchange.getRequestMethod().equals("POST")) {
                exchange.sendResponseHeaders(405, -1); return;
            }
            setCors(exchange);
            Map<String, String> params = parseQuery(exchange.getRequestURI().getQuery());
            double alpha   = Double.parseDouble(params.getOrDefault("alpha",  "0.1"));
            int maxEpochs  = Integer.parseInt(params.getOrDefault("epochs", "100"));
            double split   = Double.parseDouble(params.getOrDefault("split",  "0.8"));

            int trainN = (int)(DATASET.length * split);
            double[][] trainX = new double[trainN][3];
            int[]      trainY = new int[trainN];
            double[][] testX  = new double[DATASET.length - trainN][3];
            int[]      testY  = new int[DATASET.length - trainN];
            for (int i = 0; i < DATASET.length; i++) {
                double[] x = Perceptron.normalize(DATASET[i][0], DATASET[i][1], DATASET[i][2]);
                if (i < trainN) { trainX[i]=x; trainY[i]=DATASET[i][3]; }
                else            { testX[i-trainN]=x; testY[i-trainN]=DATASET[i][3]; }
            }
            perceptron = new Perceptron(alpha);
            int[] errs = perceptron.fit(trainX, trainY, maxEpochs);
            double trainAcc = perceptron.accuracy(trainX, trainY);
            double testAcc  = perceptron.accuracy(testX,  testY);
            double[] w = perceptron.getWeights();
            StringBuilder sb = new StringBuilder("{");
            sb.append("\"w1\":").append(w[0]).append(",\"w2\":").append(w[1])
                    .append(",\"w3\":").append(w[2]).append(",\"theta\":").append(perceptron.getTheta())
                    .append(",\"trainAcc\":").append(trainAcc).append(",\"testAcc\":").append(testAcc)
                    .append(",\"epochs\":").append(errs.length).append(",\"errors\":[");
            for (int i=0;i<errs.length;i++){if(i>0)sb.append(",");sb.append(errs[i]);}
            sb.append("]}");
            sendJson(exchange, sb.toString());
        }
    }

    // ── /predict ───────────────────────────────────────────────────────────
    static class PredictHandler implements HttpHandler {
        public void handle(HttpExchange exchange) throws IOException {
            setCors(exchange);
            if (perceptron == null) { sendJson(exchange,"{\"error\":\"Train first\"}"); return; }
            Map<String,String> p = parseQuery(exchange.getRequestURI().getQuery());
            int moisture=Integer.parseInt(p.get("moisture")),
                    lastWatered=Integer.parseInt(p.get("lastWatered")),
                    plantType=Integer.parseInt(p.get("plantType"));
            double[] x = Perceptron.normalize(moisture, lastWatered, plantType);
            int pred = perceptron.predict(x);
            double[] w = perceptron.getWeights();
            double X = x[0]*w[0]+x[1]*w[1]+x[2]*w[2]-perceptron.getTheta();
            sendJson(exchange,"{\"prediction\":"+pred+",\"X\":"+X+"}");
        }
    }

    // ── /sa  (POST, kept for compatibility) ────────────────────────────────
    static class SAHandler implements HttpHandler {
        public void handle(HttpExchange exchange) throws IOException {
            setCors(exchange);
            if (perceptron == null) { sendJson(exchange,"{\"error\":\"Train first\"}"); return; }
            String body = new String(exchange.getRequestBody().readAllBytes());
            Map<String,String> top = parseSimpleJson(body);
            int    numToWater=Integer.parseInt(top.getOrDefault("numToWater","3"));
            double T0=Double.parseDouble(top.getOrDefault("T0","200"));
            double saAlpha=Double.parseDouble(top.getOrDefault("alpha","0.97"));
            int    maxIter=Integer.parseInt(top.getOrDefault("maxIter","1000"));
            double canvasW=Double.parseDouble(top.getOrDefault("canvasW","560"));
            double canvasH=Double.parseDouble(top.getOrDefault("canvasH","380"));
            List<Plant> plantList=parsePlants(body);
            Plant[] plants=plantList.toArray(new Plant[0]);
            boolean[] needsWater=new boolean[plants.length];
            for(int i=0;i<plants.length;i++){
                double[] x=Perceptron.normalize(plants[i].moisture,plants[i].lastWatered,plants[i].plantType);
                needsWater[i]=perceptron.predict(x)==1;
            }
            double maxDist=Math.sqrt(canvasW*canvasW+canvasH*canvasH);
            SimulatedAnnealing sa=new SimulatedAnnealing(T0,saAlpha,maxIter);
            SimulatedAnnealing.Result result=sa.optimize(plants,needsWater,Math.min(numToWater,plants.length),maxDist);
            StringBuilder sb=new StringBuilder("{\"order\":[");
            for(int i=0;i<result.order.length;i++){if(i>0)sb.append(",");sb.append(result.order[i]);}
            sb.append("],\"predictions\":[");
            for(int i=0;i<needsWater.length;i++){if(i>0)sb.append(",");sb.append(needsWater[i]?1:0);}
            sb.append("],\"costHistory\":[");
            for(int i=0;i<result.costHistory.length;i+=10){if(i>0)sb.append(",");sb.append(result.costHistory[i]);}
            sb.append("]}");
            sendJson(exchange,sb.toString());
        }
    }

    // ── shared utils ───────────────────────────────────────────────────────
    static void setCors(HttpExchange e) throws IOException {
        e.getResponseHeaders().add("Access-Control-Allow-Origin","*");
        e.getResponseHeaders().add("Content-Type","application/json");
    }

    static void sendJson(HttpExchange e, String json) throws IOException {
        byte[] bytes=json.getBytes();
        e.sendResponseHeaders(200,bytes.length);
        e.getResponseBody().write(bytes);
        e.getResponseBody().close();
    }

    static Map<String,String> parseQuery(String query) {
        Map<String,String> map=new HashMap<>();
        if(query==null) return map;
        for(String part:query.split("&")){
            String[] kv=part.split("=");
            if(kv.length==2) map.put(kv[0],kv[1]);
        }
        return map;
    }

    static Map<String,String> parseSimpleJson(String json) {
        Map<String,String> map=new HashMap<>();
        for(String match:json.split(",")){
            if(match.contains("\"plants\"")||match.contains("{\"name\"")) continue;
            match=match.replaceAll("[{}\\[\\]]","").trim();
            String[] kv=match.split(":");
            if(kv.length==2){
                String k=kv[0].replaceAll("\"","").trim();
                String v=kv[1].replaceAll("\"","").trim();
                map.put(k,v);
            }
        }
        return map;
    }

    static List<Plant> parsePlants(String json) {
        List<Plant> list=new ArrayList<>();
        int start=json.indexOf("\"plants\":[");
        if(start<0) return list;
        int arrStart=json.indexOf('[',start);
        int arrEnd=json.indexOf(']',arrStart);
        String arr=json.substring(arrStart+1,arrEnd);
        for(String obj:arr.split("\\},\\{")){
            obj=obj.replaceAll("[{}]","");
            Map<String,String> fields=new HashMap<>();
            for(String part:obj.split(",")){
                String[] kv=part.split(":");
                if(kv.length==2){
                    fields.put(kv[0].replaceAll("\"","").trim(),
                            kv[1].replaceAll("\"","").trim());
                }
            }
            list.add(new Plant(
                    fields.getOrDefault("name","Plant").replaceAll("\"",""),
                    Integer.parseInt(fields.getOrDefault("x","0")),
                    Integer.parseInt(fields.getOrDefault("y","0")),
                    Integer.parseInt(fields.getOrDefault("moisture","50")),
                    Integer.parseInt(fields.getOrDefault("lastWatered","24")),
                    Integer.parseInt(fields.getOrDefault("plantType","0"))
            ));
        }
        return list;
    }
}