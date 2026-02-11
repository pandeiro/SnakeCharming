# **Project Assignment: Fine-Tuning a Language Model to Generate Recipes**

## **Project Overview**

**Objective:** Learn the fundamentals of fine-tuning by taking a pre-trained language model and specializing it to generate coherent cooking recipes. You'll observe how a model's behavior changes dramatically when trained on domain-specific data—a core technique used throughout modern AI development.

**Why This Project?**
1. **Clear Before/After Demonstration:** Watch a generic text generator become a recipe expert
2. **Hands-On ML Pipeline:** Experience data preparation, training, and evaluation
3. **Accessible Scale:** Complete on Google Colab's free tier in under 30 minutes
4. **Foundation Skill:** Fine-tuning is how ChatGPT, Claude, and specialized AI tools are created

**Prerequisites:** Basic Python (variables, loops, functions). No machine learning experience needed.

**Tools:** Google Colab (colab.research.google.com) with free T4 GPU

**Estimated Time:** 6-10 hours total (can be spread over multiple sessions)

---

## **Pedagogical Objectives**

By completing this project, you will learn to:
1. **Understand** what fine-tuning is and why it matters
2. **Prepare** datasets for machine learning
3. **Load** and configure pre-trained models
4. **Train** a model using modern libraries
5. **Evaluate** model behavior before and after training
6. **Apply** fine-tuning to real-world problems

---

## **Stage 1: Loading and Testing a Pre-Trained Model**

### **Learning Focus: Understanding Base Model Behavior**
*Core Skills: HuggingFace Transformers, model loading, text generation*

**Time Estimate:** 1-2 hours

**What You'll Build:** A baseline test showing what the model generates *before* fine-tuning. This establishes the "before" picture.

**Assignment:**
1. Create a new Colab notebook titled "Recipe_Model_Finetuning"
2. Enable GPU: Runtime → Change runtime type → Hardware accelerator → T4 GPU
3. Test what a general-purpose language model generates when asked for recipes

---

### **Step-by-Step Instructions:**

**Step 1: Install Required Libraries**

```python
# Install transformers library for working with language models
!pip install transformers datasets accelerate -q
```

**What this does:**
- `transformers`: HuggingFace's library for pre-trained models
- `datasets`: Tools for loading and processing training data
- `accelerate`: Optimizes training for GPUs

---

**Step 2: Import Libraries**

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

**Key Concepts:**
- `torch`: PyTorch, the deep learning framework
- `device`: Determines if we're using GPU (fast) or CPU (slow)
- We expect to see "Using device: cuda" if GPU is enabled

---

**Step 3: Load the Base Model**

```python
# Load pre-trained DistilGPT2 (a small, fast GPT-2 variant)
model_name = "distilgpt2"

# Fill in the loading commands:
tokenizer = ???.from_pretrained(model_name)
model = ???.from_pretrained(model_name)
model = model.to(device)

print(f"✅ Loaded {model_name}")
print(f"Model has {model.num_parameters():,} parameters")
<!-- PARTIAL_REVEAL -->
# Complete solution:
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model = model.to(device)

print(f"✅ Loaded {model_name}")
print(f"Model has {model.num_parameters():,} parameters")
```

**What's happening:**
- **Tokenizer**: Converts text → numbers the model understands
- **Model**: The neural network that generates text
- **`.to(device)`**: Moves model to GPU for faster processing
- **82 million parameters**: This is a *small* model by modern standards (GPT-4 has ~1.7 trillion!)

---

**Step 4: Create a Text Generation Function**

```python
def generate_text(prompt, max_length=150, temperature=0.8):
    """
    Generate text given a starting prompt.
    
    Parameters:
    - prompt: The starting text
    - max_length: Maximum length of generated text
    - temperature: Randomness (higher = more creative, lower = more focused)
    """
    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode back to text
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated
```

**Understanding the parameters:**
- `temperature`: Controls randomness (0.5 = focused, 1.5 = creative/chaotic)
- `top_p=0.9`: Nucleus sampling—keeps quality high while allowing variety
- `do_sample=True`: Enables random sampling (vs always picking most likely word)

---

**Step 5: Test the Base Model**

```python
# Test prompts
recipe_prompts = [
    "Recipe for chocolate chip cookies:\nIngredients:",
    "How to make pasta carbonara:\n",
    "Homemade pizza dough recipe:\nStep 1:"
]

print("=" * 60)
print("BASE MODEL OUTPUT (before fine-tuning)")
print("=" * 60)

for prompt in recipe_prompts:
    print(f"\n📝 PROMPT: {prompt}")
    print(f"🤖 GENERATED:")
    output = generate_text(prompt, max_length=120)
    print(output)
    print("-" * 60)
```

---

### **Checkpoint #1**

**What you should see:** The model will generate *something*, but it likely:
- Goes off-topic quickly
- Doesn't follow recipe structure
- Produces incoherent or generic text
- May hallucinate random facts

**Example of typical base model output:**
```
Recipe for chocolate chip cookies:
Ingredients: 2 cups of flour, 1 tablespoon of baking powder, and the following ingredients from the recipe. 
The ingredients are: 1/2 cup brown sugar, 3 tablespoons melted butter, and a few other things I can't remember right now...
```

*(Vague, incomplete, doesn't sound like a real recipe)*

**Save this output!** You'll compare it to the fine-tuned version later.

---

### **Common Issues & Debugging**

**Problem:** "RuntimeError: CUDA out of memory"
- **Solution:** Restart runtime (Runtime → Restart runtime), ensure GPU is selected

**Problem:** "No module named 'transformers'"
- **Solution:** Re-run the `!pip install` cell

**Problem:** Output is gibberish
- **Solution:** This is normal for the base model! That's why we're fine-tuning it.

---

## **Stage 2: Preparing the Recipe Dataset**

### **Learning Focus: Data Preparation for ML**
*Core Skills: Dataset loading, text formatting, train/test splits*

**Time Estimate:** 1-2 hours

**What You'll Build:** A clean, formatted dataset of recipes ready for training.

---

### **Step 1: Load Recipe Dataset**

```python
from datasets import load_dataset

# Load RecipeNLG dataset (huge collection of recipes)
print("Loading dataset... (this may take a minute)")
dataset = load_dataset("recipe_nlg", split="train")

# Let's look at what we have
print(f"Total recipes: {len(dataset):,}")
print(f"\nSample recipe:")
print(dataset[0])
```

**What you should see:** A dictionary with fields like:
- `title`: Recipe name
- `ingredients`: List of ingredients
- `directions`: Cooking instructions

---

### **Step 2: Format Recipes for Training**

We need to convert recipes into a consistent text format the model can learn from.

```python
def format_recipe(example):
    """
    Convert a recipe dictionary into formatted training text.
    
    Format:
    RECIPE: [Title]
    INGREDIENTS: [ingredients list]
    DIRECTIONS: [directions]
    [END]
    """
    # Fill in the formatting logic:
    title = example['???']
    ingredients = ', '.join(example['???'])
    directions = example['???']
    
    formatted = f"""RECIPE: {title}
INGREDIENTS: {ingredients}
DIRECTIONS: {directions}
[END]"""
    
    return {"text": formatted}
<!-- PARTIAL_REVEAL -->
# Complete solution:
def format_recipe(example):
    """
    Convert a recipe dictionary into formatted training text.
    """
    title = example['title']
    
    # Handle ingredients (might be string or list)
    if isinstance(example['ingredients'], list):
        ingredients = ', '.join(example['ingredients'])
    else:
        ingredients = example['ingredients']
    
    directions = example['directions']
    
    formatted = f"""RECIPE: {title}
INGREDIENTS: {ingredients}
DIRECTIONS: {directions}
[END]"""
    
    return {"text": formatted}

# Test the formatter
print(format_recipe(dataset[0])['text'][:300])
```

**Why this format?**
- **Consistent structure** helps the model learn patterns
- **Clear markers** (RECIPE, INGREDIENTS, DIRECTIONS) give the model cues
- **[END] token** tells the model when to stop generating

---

**Step 3: Process and Split the Dataset**

```python
# We'll use a subset for faster training (full dataset would take hours)
# Select 5000 recipes for training, 500 for validation
small_dataset = dataset.select(range(5500))

# Apply formatting to all recipes
formatted_dataset = small_dataset.map(
    format_recipe,
    remove_columns=small_dataset.column_names
)

# Split into training and validation sets
# Fill in the split ratio:
train_test = formatted_dataset.train_test_split(test_size=???)  # What fraction for testing?

train_dataset = train_test['train']
val_dataset = train_test['test']

print(f"✅ Training recipes: {len(train_dataset):,}")
print(f"✅ Validation recipes: {len(val_dataset):,}")
<!-- PARTIAL_REVEAL -->
# Complete solution:
train_test = formatted_dataset.train_test_split(test_size=0.1)  # 10% for validation

train_dataset = train_test['train']
val_dataset = train_test['test']

print(f"✅ Training recipes: {len(train_dataset):,}")
print(f"✅ Validation recipes: {len(val_dataset):,}")
```

**Understanding train/test split:**
- **Training set** (90%): Model learns from these
- **Validation set** (10%): Tests if model generalizes to new recipes
- Never test on training data—that's cheating!

---

### **Checkpoint #2**

**Verify your data:**
```python
# Look at a formatted training example
print("Sample formatted recipe:")
print(train_dataset[0]['text'])
```

**You should see:** A cleanly formatted recipe with clear structure.

---

## **Stage 3: Tokenizing the Data**

### **Learning Focus: How Models Process Text**
*Core Skills: Tokenization, padding, attention masks*

**Time Estimate:** 1 hour

**What You'll Build:** Convert text recipes into numerical tensors the model can train on.

---

### **Step 1: Configure the Tokenizer**

```python
# Set padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    """
    Convert text to tokens (numbers) with fixed length.
    """
    # Tokenize with truncation and padding
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,  # Maximum sequence length
        padding='max_length'
    )

# Apply tokenization to entire dataset
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

print("✅ Data tokenized and ready for training")
```

**What's happening:**
- **Tokenization**: Text → numbers (e.g., "cookie" → 14523)
- **Truncation**: Cuts recipes longer than 512 tokens
- **Padding**: Makes all sequences same length (needed for batching)

---

### **Checkpoint #3**

```python
# Inspect tokenized data
print("Original text:")
print(train_dataset[0]['text'][:200])
print("\nTokenized (first 20 tokens):")
print(tokenized_train[0]['input_ids'][:20])
print(f"\nTotal tokens in this recipe: {len(tokenized_train[0]['input_ids'])}")
```

**What to expect:** A list of numbers representing the text.

---

## **Stage 4: Fine-Tuning the Model**

### **Learning Focus: Training a Neural Network**
*Core Skills: Training loops, loss functions, optimization, checkpointing*

**Time Estimate:** 2-3 hours (mostly waiting for training)

**What You'll Build:** Actually train the model to generate recipe text.

---

### **Step 1: Set Up Training Configuration**

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Set up data collator (handles batching)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked
)

# Configure training parameters
# Fill in reasonable values:
training_args = TrainingArguments(
    output_dir="./recipe-model",
    num_train_epochs=???,           # How many times to go through the data? Try 3
    per_device_train_batch_size=???, # Batch size (try 4-8)
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=???               # Learning rate (try 5e-5)
)
<!-- PARTIAL_REVEAL -->
# Complete solution:
training_args = TrainingArguments(
    output_dir="./recipe-model",
    num_train_epochs=3,                    # 3 passes through data
    per_device_train_batch_size=4,         # 4 recipes per batch
    per_device_eval_batch_size=4,
    save_steps=500,                        # Save checkpoint every 500 steps
    save_total_limit=2,                    # Keep only 2 checkpoints
    logging_steps=100,                     # Log every 100 steps
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=5e-5,                    # Standard learning rate
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
)
```

**Understanding training parameters:**
- **Epochs**: How many times to see the entire dataset (more ≠ always better)
- **Batch size**: Recipes processed simultaneously (limited by GPU memory)
- **Learning rate**: How much to adjust weights each step (5e-5 = 0.00005)
- **Warmup**: Gradually increases learning rate at start (prevents instability)

---

### **Step 2: Create the Trainer**

```python
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

print("✅ Trainer configured and ready")
```

---

### **Step 3: Start Training!**

```python
# This is where the magic happens
print("🚀 Starting training...")
print("This will take ~10-15 minutes on a T4 GPU")
print("-" * 60)

trainer.train()

print("✅ Training complete!")
```

**What you'll see during training:**
- Loss decreasing (good! model is learning)
- Progress bar showing steps
- Periodic evaluation results
- GPU memory usage

**Expected behavior:**
- **Initial loss**: ~3.5-4.0
- **Final loss**: ~1.5-2.0 (lower is better)
- If loss increases or stays flat, something's wrong

---

### **Step 4: Save the Fine-Tuned Model**

```python
# Save model and tokenizer
model.save_pretrained("./recipe-gpt2-finetuned")
tokenizer.save_pretrained("./recipe-gpt2-finetuned")

print("✅ Model saved to ./recipe-gpt2-finetuned")
```

---

### **Checkpoint #4**

**Verification checklist:**
- ✅ Training completed without errors
- ✅ Loss decreased over time
- ✅ Model saved successfully
- ✅ Total training time was reasonable (~10-20 min)

---

### **Common Training Issues**

**Problem:** "CUDA out of memory"
- **Solution:** Reduce `per_device_train_batch_size` to 2 or 1

**Problem:** Loss isn't decreasing
- **Solution:** Check learning rate (try 1e-5 to 1e-4), verify data is formatted correctly

**Problem:** Training is very slow
- **Solution:** Confirm GPU is enabled (should see "cuda" in device check)

**Problem:** Model predicts same thing repeatedly
- **Solution:** Normal during training; will improve by final epoch

---

## **Stage 5: Testing the Fine-Tuned Model**

### **Learning Focus: Evaluating Model Performance**
*Core Skills: Qualitative evaluation, before/after comparison*

**Time Estimate:** 1 hour

**What You'll Build:** A side-by-side comparison showing how fine-tuning changed behavior.

---

### **Step 1: Load the Fine-Tuned Model**

```python
# Load the fine-tuned version
finetuned_model = GPT2LMHeadModel.from_pretrained("./recipe-gpt2-finetuned")
finetuned_model = finetuned_model.to(device)

print("✅ Fine-tuned model loaded")
```

---

### **Step 2: Create Comparison Function**

```python
def compare_models(prompt, base_model, finetuned_model, max_length=200):
    """
    Generate text from both base and fine-tuned models for comparison.
    """
    print("=" * 70)
    print(f"PROMPT: {prompt}")
    print("=" * 70)
    
    # Generate from base model
    print("\n🔵 BASE MODEL (before fine-tuning):")
    print("-" * 70)
    base_input = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        base_output = base_model.generate(
            base_input,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)
    print(base_text)
    
    # Generate from fine-tuned model
    print("\n🟢 FINE-TUNED MODEL (after training on recipes):")
    print("-" * 70)
    ft_input = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ft_output = finetuned_model.generate(
            ft_input,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    ft_text = tokenizer.decode(ft_output[0], skip_special_tokens=True)
    print(ft_text)
    print("\n" + "=" * 70 + "\n")
```

---

### **Step 3: Test with Various Prompts**

```python
# Test prompts
test_prompts = [
    "RECIPE: Chocolate Chip Cookies\nINGREDIENTS:",
    "RECIPE: Spaghetti Carbonara\nINGREDIENTS:",
    "RECIPE: Homemade Pizza\nINGREDIENTS:",
    "RECIPE: Chicken Stir Fry\nINGREDIENTS:",
]

# Run comparisons
for prompt in test_prompts:
    compare_models(prompt, model, finetuned_model, max_length=250)
```

---

### **Checkpoint #5**

**What you should observe:**

**Base Model:** 
- Inconsistent format
- Goes off-topic
- Vague or nonsensical ingredients
- Doesn't follow recipe structure

**Fine-Tuned Model:**
- Follows RECIPE/INGREDIENTS/DIRECTIONS format
- Lists plausible ingredients
- Generates coherent cooking steps
- Stays on-topic

**Example comparison:**

```
PROMPT: RECIPE: Chocolate Chip Cookies
INGREDIENTS:

BASE MODEL:
...might output generic text about cookies or random content...

FINE-TUNED MODEL:
2 cups all-purpose flour, 1 teaspoon baking soda, 1 teaspoon salt, 
1 cup butter softened, 3/4 cup granulated sugar, 3/4 cup packed brown sugar,
2 eggs, 2 teaspoons vanilla extract, 2 cups chocolate chips
DIRECTIONS: Preheat oven to 375°F. Mix dry ingredients in bowl...
```

---

### **Analysis Questions**

Create a markdown cell and answer:

1. **What changed?** Describe 3 specific differences between base and fine-tuned outputs.

2. **Why does fine-tuning work?** In your own words, explain why the model got better at recipes.

3. **What didn't change?** The model still has the same architecture and parameters—what did training actually modify?

---

## **Stage 6: Experimenting with Generation Parameters**

### **Learning Focus: Controlling Model Behavior**
*Core Skills: Hyperparameter tuning, temperature, top-p sampling*

**Time Estimate:** 1 hour

**What You'll Explore:** How different generation settings affect output quality and creativity.

---

### **Step 1: Temperature Comparison**

```python
def test_temperatures(prompt, model, temperatures=[0.3, 0.7, 1.0, 1.5]):
    """
    Generate text with different temperature values.
    """
    print(f"PROMPT: {prompt}\n")
    
    for temp in temperatures:
        print(f"🌡️ Temperature = {temp}")
        print("-" * 60)
        
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=150,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(text)
        print("\n")

# Test it
test_temperatures(
    "RECIPE: Banana Bread\nINGREDIENTS:",
    finetuned_model
)
```

**What to observe:**
- **Low temp (0.3)**: More conservative, repetitive, "safe" choices
- **Medium temp (0.7)**: Good balance of coherence and variety
- **High temp (1.5)**: More creative but potentially nonsensical

---

### **Step 2: Create Your Own Recipe Generator**

```python
def generate_custom_recipe(dish_name, temperature=0.8):
    """
    Generate a complete recipe for any dish.
    """
    prompt = f"RECIPE: {dish_name}\nINGREDIENTS:"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = finetuned_model.generate(
            inputs,
            max_length=300,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return recipe

# Try your own recipes!
my_dishes = [
    "Garlic Butter Shrimp",
    "Vegetarian Tacos",
    "Lemon Cake"
]

for dish in my_dishes:
    print("=" * 70)
    print(generate_custom_recipe(dish))
    print("\n")
```

---

## **Project Deliverables**

Submit your Colab notebook with:

### **1. Code (Well-Commented)**
- All stages completed and working
- Clear comments explaining each section
- Organized with markdown headers

### **2. Outputs**
Include:
- Base model generation examples (Stage 1)
- Fine-tuned model generation examples (Stage 5)
- Side-by-side comparisons
- Temperature experiment results (Stage 6)

### **3. Written Responses**

Answer these questions in markdown cells:

**Understanding Questions:**
1. What is fine-tuning? Explain in 2-3 sentences to someone who doesn't know ML.
2. Why did we use only 5,000 recipes instead of the full dataset (2 million+)?
3. What would happen if we trained for 20 epochs instead of 3?

**Analysis Questions:**
4. Compare the base vs fine-tuned model:
   - Which follows recipe structure better?
   - Which stays on-topic more consistently?
   - Which would you trust for actual cooking?

5. Based on your temperature experiments:
   - What temperature gave the best results?
   - When might you want high temperature? Low temperature?
   - What's the tradeoff?

**Application Questions:**
6. Name three other domains where fine-tuning could be useful:
   - Example: "Medical diagnosis from patient notes"
   - Your ideas: ???

7. If you wanted to fine-tune a model to write Python code, what would you need?

**Reflection:**
8. What surprised you most about this project?
9. What was challenging about preparing the dataset?
10. How is this different from "training a model from scratch"?

---

## **Success Criteria**

### **Complete (Meets Expectations)**
- ✅ Stages 1-5 fully functional
- ✅ Base model output captured
- ✅ Fine-tuning completed successfully
- ✅ Clear before/after comparison
- ✅ Analysis questions answered

### **Proficient (Exceeds Expectations)**
- ✅ All of above, plus:
- ✅ Temperature experiments completed (Stage 6)
- ✅ Code is well-documented
- ✅ Thoughtful analysis of results
- ✅ Tested with creative/unusual prompts

### **Excellent (Outstanding)**
- ✅ All of above, plus:
- ✅ Trained for different epoch counts and compared
- ✅ Experimented with different dataset sizes
- ✅ Analyzed failure cases (when model still fails)
- ✅ Clear documentation that could teach others

### **Exceptional (Goes Above and Beyond)**
- ✅ All of above, plus:
- ✅ Fine-tuned on a different dataset (your choice!)
- ✅ Built a simple web interface or interactive tool
- ✅ Compared DistilGPT2 to a different base model
- ✅ Created evaluation metrics (not just qualitative)

---

## **Tips for Success**

### **Before You Start**
- Read the entire assignment to understand the flow
- Ensure GPU is enabled (Runtime → Change runtime type → T4 GPU)
- Budget 30 minutes for training—don't start right before you need to leave

### **While Working**
- Save frequently (File → Save or Ctrl+S)
- Run cells in order—don't skip ahead
- If training fails, check GPU memory and reduce batch size
- Keep a notes section for observations

### **When You Get Stuck**
1. **Check error messages** carefully
2. **Verify GPU** is enabled and being used
3. **Restart runtime** if things get corrupted
4. **Reduce batch size** if you see CUDA memory errors
5. **Ask for help** with specific error messages

### **Best Practices**
- Comment your code as you write
- Use markdown cells to explain your thinking
- Save before starting training (sessions can crash)
- Test on simple prompts first before complex ones

---

## **Extensions & Next Steps**

After completing this project, you could:

1. **Different Datasets:** Fine-tune on song lyrics, movie scripts, or academic papers
2. **Larger Models:** Try GPT-2 (124M params) or GPT-2-medium (355M) if you have Pro Colab
3. **Instruction Following:** Fine-tune on instruction-following datasets
4. **Evaluation Metrics:** Implement perplexity or BLEU scores
5. **Multi-Task:** Combine recipes + other domains in one model
6. **Deployment:** Save model and create a simple Gradio interface

**Want to go deeper?**
- Learn about LoRA (parameter-efficient fine-tuning)
- Explore RLHF (reinforcement learning from human feedback)
- Try fine-tuning vision models or multimodal models
- Study the Hugging Face course on fine-tuning

---

## **Key Takeaways**

**What is fine-tuning?**
- Taking a pre-trained model and specializing it for a specific task
- Adjusts weights slightly to fit new data patterns
- Much faster and cheaper than training from scratch

**Why does it matter?**
- How ChatGPT became helpful (started as base GPT, fine-tuned on instructions)
- How medical AI, legal AI, coding assistants are built
- Core technique in modern AI development

**What did you learn?**
- Models are adaptable through additional training
- Data quality and format matter enormously
- Trade-offs exist (epochs, batch size, temperature)
- Even small models can be specialized effectively

**You've experienced the foundation of how modern AI systems are created! 🚀**
