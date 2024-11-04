def plot_training_curves(history, history_finetuning):
    fig, axes = plt.subplots(2, 3, figsize=(20, 9), dpi=100)
    
    metric_keys = ["auc", "acc", "loss"]
    metric_names = ["auc", "acc", "loss"]
    
    # Plotting for MODEL III Training
    for i, (metric_key, metric_name) in enumerate(zip(metric_keys, metric_names)):
        axes[0,i].plot(history.history[f"{metric_key}"], label="Supervised ResNet50 Baseline", linestyle='--', linewidth=2)
        axes[0,i].plot(history_finetuning.history[f"{metric_key}"], label="Self-Supervised Finetuning (BYOL)", linestyle='--', linewidth=2)

        axes[0,i].legend(fontsize='large')
        axes[0,i].set_title(f"MODEL II: Classification TRAIN {metric_name} during training")
        axes[0,i].set_xlabel("epochs")
        axes[0,i].set_ylabel(f"Training {metric_name}")

    # Plotting for MODEL III Validation
    for i, (metric_key, metric_name) in enumerate(zip(metric_keys, metric_names)):
        axes[1,i].plot(history.history[f"{metric_key}"], label="Supervised ResNet50 Baseline", linestyle='--', linewidth=2)
        axes[1,i].plot(history_finetuning.history[f"val_{metric_key}"], label="Self-Supervised Finetuning (BYOL)", linestyle='--', linewidth=2)

        axes[1,i].legend(fontsize='large')
        axes[1,i].set_title(f"MODEL II: Classification VALIDATION {metric_name} during training")
        axes[1,i].set_xlabel("epochs")
        axes[1,i].set_ylabel(f"Validation {metric_name}")
    
    plt.tight_layout()
    plt.show()

plot_training_curves(history, history_finetuning)

val_file_paths = []
classes = ['axion', 'cdm', 'no_sub']
class_indices = {class_name: idx for idx, class_name in enumerate(classes)}
for class_name in classes:
    val_class_dir = os.path.join('Model_II_test' , class_name)
    val_file_paths += [os.path.join(val_class_dir, file) for file in os.listdir(val_class_dir)]


num_batches = val_steps

y_true_baseline = []
y_pred_baseline = []
y_true_finetuning = []
y_pred_finetuning = []

for i in range(num_batches):
    X_batch, y_batch = next(data.validation_generator())
    y_true_baseline.extend(y_batch)
    y_pred_baseline.extend(model_supervised.predict(X_batch))
    y_true_finetuning.extend(y_batch)
    y_pred_finetuning.extend(finetuning_model.predict(X_batch))
# Here is the extra logic for handling the last smaller batch
if len(val_file_paths) % batch_size != 0:
    X_batch, y_batch = next(data.validation_generator())
    y_true_baseline.extend(y_batch)
    y_pred_baseline.extend(model_supervised.predict(X_batch))
    y_true_finetuning.extend(y_batch)
    y_pred_finetuning.extend(finetuning_model.predict(X_batch))

y_true_baseline = np.array(y_true_baseline)
y_pred_baseline = np.array(y_pred_baseline)
y_true_finetuning = np.array(y_true_finetuning)
y_pred_finetuning = np.array(y_pred_finetuning)

y_true_baseline_int = np.argmax(y_true_baseline, axis=1)
y_pred_baseline_int = np.argmax(y_pred_baseline, axis=1)
y_true_finetuning_int = np.argmax(y_true_finetuning, axis=1)
y_pred_finetuning_int = np.argmax(y_pred_finetuning, axis=1)

auc_baseline = roc_auc_score(y_true_baseline, y_pred_baseline, average='macro', multi_class='ovr')
print("Separate test set results (MODEL II test: Containing 5000 samples per class):\n")
print(f"AUC (Baseline): {int(auc_baseline * 1000) / 1000}")  
accuracy_baseline = accuracy_score(y_true_baseline_int, y_pred_baseline_int)
print("\nMODEL II: Classification Report (Baseline):")
print(classification_report(y_true_baseline_int, y_pred_baseline_int))

auc_finetuning = roc_auc_score(y_true_finetuning, y_pred_finetuning, average='macro', multi_class='ovr')
print("Separate test set results (MODEL II test: Containing 5000 samples per class):\n")
print(f"AUC (Finetuning): {int(auc_finetuning * 1000) / 1000}")  
accuracy_finetuning = accuracy_score(y_true_finetuning_int, y_pred_finetuning_int)
print("\nMODEL II: Classification Report (Finetuning):")
print(classification_report(y_true_finetuning_int, y_pred_finetuning_int))

datasets = {
    "Resnet50 Baseline": (y_true_baseline, y_pred_baseline),
    "Finetuned BYOL": (y_true_finetuning, y_pred_finetuning),
}

plt.figure(figsize=(12, 8), dpi=100) 
for idx, (title, (y_true, y_pred)) in enumerate(datasets.items()):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    plt.subplot(len(datasets), 2, 2*idx+1)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f'MODEL II: Confusion Matrix \n({title})', fontsize=12)
    plt.xlabel('Predicted Labels', fontsize=11)
    plt.ylabel('True Labels', fontsize=11)

    tick_marks = np.arange(len(class_indices))
    plt.xticks(tick_marks, class_indices)
    plt.yticks(tick_marks, class_indices, rotation=90)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.subplot(len(datasets), 2, 2*idx+2)
    for class_name, i in class_indices.items():
        plt.plot(fpr[i], tpr[i], label=f"{class_name} substructure (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f'MODEL II: ROC \n({title})', fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.legend(loc="lower right")

plt.subplots_adjust(wspace=0, hspace=0.4)
plt.show()
