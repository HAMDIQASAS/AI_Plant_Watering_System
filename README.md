# AI Plant Watering System

A smart plant-watering scheduler that combines a **Perceptron classifier** with **Simulated Annealing**.

The perceptron predicts whether each plant needs water based on its current condition, while simulated annealing searches for an efficient watering sequence that balances plant needs with walking distance.

## Features

- Train a perceptron directly from the browser
- Adjustable learning rate, epoch count, and train/test split
- Live training updates using Server-Sent Events (SSE)
- Display training accuracy, test accuracy, errors, and learned weights
- Test custom plant conditions after training
- Interactive garden canvas for placing plants
- Support for three plant types: Cactus, Flower, and Herb
- Predict which plants need watering
- Choose how many plants should be watered
- Optimize the watering order using Simulated Annealing
- Adjustable initial temperature, cooling rate, and iteration count
- Live optimization progress and cost history
- Visualize the final watering route and optimization results

## How It Works

### Perceptron Classifier

Each plant is represented using three input features:

| Feature | Description | Normalization |
| --- | --- | --- |
| Soil moisture | Current soil moisture level from 0 to 100 | `moisture / 100` |
| Last watered | Hours since the plant was last watered, from 0 to 48 | `lastWatered / 48` |
| Plant type | Cactus = 0, Flower = 1, Herb = 2 | `plantType / 2` |

The perceptron calculates:

```text
X = x1W1 + x2W2 + x3W3 - theta
```

Prediction:

```text
X >= 0  -> Needs water
X < 0   -> Does not need water
```

Weights are updated whenever a training sample is misclassified.

### Simulated Annealing

After the plants are classified, simulated annealing searches for a good watering sequence.

The optimization cost is:

```text
cost = plants_missed + normalized_distance + extra_watered
```

Where:

- `plants_missed` = plants predicted to need water but not selected
- `normalized_distance` = total Euclidean walking distance divided by the garden canvas diagonal
- `extra_watered` = selected plants predicted not to need water

The optimizer creates neighboring solutions by either swapping two plants in the route or replacing one selected plant with another.

## Project Structure

```text
AI_Plant_Watering_System-master/
├── pom.xml
├── src/
│   └── main/
│       ├── backend/
│       │   └── org/
│       │       └── example/
│       │           ├── Main.java
│       │           ├── Perceptron.java
│       │           ├── Plant.java
│       │           └── SimulatedAnnealing.java
│       └── frontend/
│           └── index.html
└── README.md
```

## Technologies

### Backend

- Java 11+
- Java built-in `HttpServer`
- Server-Sent Events (SSE)

### Frontend

- HTML5
- CSS3
- Vanilla JavaScript
- HTML Canvas

### Algorithms

- Perceptron binary classifier
- Simulated Annealing
- Euclidean-distance route optimization

No external Java libraries are required.

## Requirements

- JDK 11 or newer
- A modern web browser

Optional:

- IntelliJ IDEA or another Java IDE

## Running the Project

The Java source files are stored in `src/main/backend`, which is not Maven's standard `src/main/java` directory. The easiest way to run the project is either from an IDE or by compiling it manually.

### Option 1: Run From IntelliJ IDEA

1. Open the project.
2. Open:

```text
src/main/backend/org/example/Main.java
```

3. Run `Main.main()`.
4. The backend starts at:

```text
http://localhost:8080
```

5. Open:

```text
src/main/frontend/index.html
```

in your browser.

### Option 2: Run From the Terminal

From the project root:

```bash
mkdir -p out
javac -d out src/main/backend/org/example/*.java
java -cp out org.example.Main
```

On Windows Command Prompt:

```bat
mkdir out
javac -d out src\main\backend\org\example\*.java
java -cp out org.example.Main
```

The console should display:

```text
Server running at http://localhost:8080
Open frontend/index.html in your browser
```

Then open `src/main/frontend/index.html`.

If your browser blocks requests from a local `file://` page, serve the frontend using a simple local web server.

For example:

```bash
cd src/main/frontend
python3 -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

Keep the Java backend running on port `8080`.

## Using the Application

### 1. Review the Dataset

Open the **Training Dataset** tab to view the built-in samples.

Each sample contains:

```text
[moisture, lastWatered, plantType, label]
```

Labels:

- `1` = needs water
- `0` = does not need water

### 2. Train the Perceptron

Open the **Perceptron** tab and choose:

- learning rate
- maximum epochs
- train/test split

Then click **Train Perceptron**.

The frontend displays live training progress including errors, accuracy, and learned weights.

### 3. Test a Plant

After training, enter:

- soil moisture
- hours since last watering
- plant type

Then click **Predict**.

### 4. Build the Garden

Open the **Garden** tab.

You can:

- click the canvas to choose a plant position
- enter plant information
- add plants manually
- load a sample garden
- remove plants
- clear the garden
- choose how many plants should be watered

### 5. Optimize the Watering Route

Open the **Optimizer** tab and configure:

- Initial Temperature `T0`
- Cooling Rate `alpha`
- Maximum Iterations

Click **Run Simulated Annealing**.

The application displays the optimization process, current cost, candidate routes, and best solution found.

### 6. View Results

Open the **Results** tab to see the final selected plants, predictions, learned perceptron parameters, and optimized watering sequence.

## Backend Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/train-stream` | GET | Train the perceptron with live SSE updates |
| `/train` | POST | Train the perceptron and return JSON results |
| `/predict` | GET | Predict whether a plant needs water |
| `/sa-stream` | POST | Run simulated annealing with live SSE updates |
| `/sa` | POST | Run simulated annealing and return JSON results |

### Example Prediction Request

After training the model:

```text
GET /predict?moisture=30&lastWatered=20&plantType=1
```

Example response:

```json
{
  "prediction": 1,
  "X": 0.1234
}
```

## Important Notes

- The perceptron must be trained before prediction or optimization.
- The training dataset is hard-coded in `Main.java`.
- The trained model exists only while the Java server is running.
- Restarting the backend resets the perceptron.
- Initial perceptron weights are random, so training results may differ between runs.
- Simulated annealing also uses randomness, so the optimized route may vary.
- Port `8080` must be free before starting the backend.

## Possible Improvements

- Save and reload trained perceptron weights
- Store plants in a database
- Connect real soil-moisture sensors
- Add more plant types and environmental features
- Shuffle the dataset before the train/test split
- Update the perceptron threshold during training
- Compare the perceptron with other classifiers
- Add configurable start and end positions for the watering route
- Deploy the frontend and backend together
- Connect the scheduler to an automatic irrigation system

## License

This project currently does not include a license file.
