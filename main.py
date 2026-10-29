import math
import random
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock


# --- Pure Python Trainable Neural Network ---
class NeuralNetwork:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        self.w1 = [[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.w2 = [[random.uniform(-1, 1) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0 for _ in range(output_dim)]

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-max(min(x, 15.0), -15.0)))

    def sigmoid_deriv(self, output):
        return output * (1.0 - output)

    def forward(self, x):
        # Hidden layer
        self.hidden = [0.0] * len(self.b1)
        for j in range(len(self.b1)):
            s = sum(x[i] * self.w1[i][j] for i in range(len(x))) + self.b1[j]
            self.hidden[j] = self.sigmoid(s)
        # Output layer
        self.output = [0.0] * len(self.b2)
        for k in range(len(self.b2)):
            s = sum(self.hidden[j] * self.w2[j][k] for j in range(len(self.hidden))) + self.b2[k]
            self.output[k] = self.sigmoid(s)
        return self.output

    def train_step(self, dataset, lr=0.5):
        total_loss = 0.0
        for x, y_target in dataset:
            y_pred = self.forward(x)
            
            # Compute loss and output gradients
            out_deltas = [0.0] * len(y_pred)
            for k in range(len(y_pred)):
                error = y_target[k] - y_pred[k]
                total_loss += error ** 2
                out_deltas[k] = error * self.sigmoid_deriv(y_pred[k])

            # Compute hidden layer gradients
            hidden_deltas = [0.0] * len(self.hidden)
            for j in range(len(self.hidden)):
                error = sum(out_deltas[k] * self.w2[j][k] for k in range(len(out_deltas)))
                hidden_deltas[j] = error * self.sigmoid_deriv(self.hidden[j])

            # Update weights & biases (Gradient Descent)
            for j in range(len(self.hidden)):
                for k in range(len(out_deltas)):
                    self.w2[j][k] += lr * out_deltas[k] * self.hidden[j]
            for k in range(len(out_deltas)):
                self.b2[k] += lr * out_deltas[k]

            for i in range(len(x)):
                for j in range(len(hidden_deltas)):
                    self.w1[i][j] += lr * hidden_deltas[j] * x[i]
            for j in range(len(hidden_deltas)):
                self.b1[j] += lr * hidden_deltas[j]

        return total_loss / len(dataset)


# --- Mobile User Interface ---
class TrainAILocalApp(App):
    def build(self):
        self.nn = NeuralNetwork()
        self.dataset = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ]
        self.epoch = 0

        layout = BoxLayout(orientation='vertical', padding=20, spacing=15)

        self.title_label = Label(
            text="[b]On-Device AI Trainer[/b]\nTask: Learn XOR Gate",
            markup=True,
            font_size="20sp",
            halign="center"
        )
        layout.add_widget(self.title_label)

        self.status_label = Label(
            text="Model Status: Untrained (Random Weights)\nEpoch: 0 | Loss: --",
            font_size="15sp",
            halign="center"
        )
        layout.add_widget(self.status_label)

        self.train_btn = Button(
            text="Train AI on Phone (+500 Epochs)",
            size_hint=(1, 0.25),
            background_color=(0.1, 0.6, 0.9, 1)
        )
        self.train_btn.bind(on_press=self.train_model)
        layout.add_widget(self.train_btn)

        # Prediction UI
        test_box = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, 0.2))
        self.in_a = TextInput(text="1", multiline=False, font_size="18sp")
        self.in_b = TextInput(text="0", multiline=False, font_size="18sp")
        test_box.add_widget(self.in_a)
        test_box.add_widget(self.in_b)
        layout.add_widget(test_box)

        self.predict_btn = Button(
            text="Run Prediction",
            size_hint=(1, 0.25),
            background_color=(0.2, 0.8, 0.3, 1)
        )
        self.predict_btn.bind(on_press=self.predict)
        layout.add_widget(self.predict_btn)

        self.result_label = Label(
            text="Prediction: Press button",
            font_size="16sp"
        )
        layout.add_widget(self.result_label)

        return layout

    def train_model(self, instance):
        for _ in range(500):
            loss = self.nn.train_step(self.dataset, lr=0.6)
            self.epoch += 1

        self.status_label.text = f"Trained locally on device!\nEpochs: {self.epoch} | Loss: {loss:.5f}"

    def predict(self, instance):
        try:
            val_a = float(self.in_a.text.strip())
            val_b = float(self.in_b.text.strip())
            pred = self.nn.forward([val_a, val_b])[0]
            rounded = 1 if pred >= 0.5 else 0
            self.result_label.text = f"Raw: {pred:.4f} -> Predicted Class: {rounded}"
        except Exception as e:
            self.result_label.text = "Error: Input 0 or 1 numbers"


if __name__ == "__main__":
    TrainAILocalApp().run()
