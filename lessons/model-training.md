# **Project Assignment: Fine-Tuning a Language Model to Generate Recipes**

## **Project Overview**

**Objective:** Learn the fundamentals of fine-tuning by taking a pre-trained language model and specializing it to generate coherent cooking recipes. You'll observe how a model's behavior changes dramatically when trained on domain-specific data—a core technique used throughout modern AI development.

**Why This Project?**
1. **Clear Before/After Demonstration:** Watch a generic text generator become a recipe expert
2. **Hands-On ML Pipeline:** Experience data preparation, training, and evaluation
3. **Accessible Scale:** Complete on Google Colab's free tier in under 30 minutes
4. **Foundation Skill:** Fine-tuning is how ChatGPT, Claude, and specialized AI tools are created
5. **Modern Technique:** Learn LoRA (Low-Rank Adaptation), the industry-standard approach

**Prerequisites:** Basic Python (variables, loops, functions). No machine learning experience needed.

**Tools:** Google Colab (colab.research.google.com) with free T4 GPU

**Estimated Time:** 6-10 hours total (can be spread over multiple sessions)

---

## **Pedagogical Objectives**

By completing this project, you will learn to:
1. **Understand** what fine-tuning is and what actually changes inside the model
2. **Prepare** messy real-world datasets for machine learning
3. **Configure** modern parameter-efficient fine-tuning (LoRA)
4. **Train** a model and understand what's happening during training
5. **Evaluate** model behavior quantitatively and qualitatively
6. **Apply** fine-tuning responsibly and understand its limitations

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
!pip install transformers datasets accelerate peft -q
```

**What this does:**
- `transformers`: HuggingFace's library for pre-trained models
- `datasets`: Tools for loading and processing training data
- `accelerate`: Optimizes training for GPUs
- `peft`: Parameter-Efficient Fine-Tuning (for LoRA)

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

# Verify GPU
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**Key Concepts:**
- `torch`: PyTorch, the deep learning framework
- `device`: Determines if we're using GPU (fast) or CPU (slow)
- We expect to see "Using device: cuda" if GPU is enabled

**If you see "cpu":** Go to Runtime → Change runtime type → Select T4 GPU

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

**Step 4: Understanding Tokenization**

Before we generate text, let's understand how models see language:

```python
# Tokens are NOT words!
# Let's see how text gets broken down:

sample_text = "chocolate chip cookies"
tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.encode(sample_text)

print(f"Original text: '{sample_text}'")
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
print(f"\nNotice: '{sample_text}' became {len(tokens)} tokens")

# Try more examples
for phrase in ["chocolate", "chocolatey-ness", "supercalifragilisticexpialidocious"]:
    tokens = tokenizer.tokenize(phrase)
    print(f"'{phrase}' → {tokens} ({len(tokens)} tokens)")
```

**Key Insight:** 
- Tokens are **subword units**, not words
- "chocolate" might be 1 token
- "chocolatey-ness" might be 3 tokens
- This is how models handle rare words and typos

**Why this matters:** The model doesn't "read" text like you do. It sees sequences of numbers representing token chunks.

---

**Step 5: Create a Text Generation Function**

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

**Step 6: Test the Base Model**

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
- Doesn't understand what a recipe is supposed to look like

**Example of typical base model output:**
```
Recipe for chocolate chip cookies:
Ingredients: 2 cups of flour, 1 tablespoon of baking powder, and the following ingredients from the recipe. 
The ingredients are: 1/2 cup brown sugar, 3 tablespoons melted butter, and a few other things I can't remember right now...
```

*(Vague, incomplete, doesn't sound like a real recipe)*

**🔴 IMPORTANT:** Save this output! You'll compare it to the fine-tuned version later.

---

## **Conceptual Interlude: What IS Fine-Tuning?**

### **What Actually Happens Inside the Model**

Before we continue, let's understand what we're about to do.

**The Model's Brain:**
- The model contains **82 million numerical parameters** (also called "weights")
- Each parameter is like a tiny volume knob
- Together, they encode probability distributions: "What word likely comes next?"

**What Fine-Tuning Does:**
Fine-tuning is **continuing training on new text**, so these billions of little numeric knobs shift slightly toward your domain.

**What Fine-Tuning Is NOT:**
- ❌ Installing "recipe mode"
- ❌ Adding new rules or logic
- ❌ Teaching the model what food tastes like
- ❌ Giving it memory or reasoning

**What Fine-Tuning IS:**
- ✅ Nudging probability space
- ✅ Making recipe-shaped text more probable
- ✅ Adjusting internal statistics

**The Miracle:**
After training on recipes, the model's internal statistics now say:
> "After INGREDIENTS: comes a list of food nouns with high probability, not a political rant or random story."

That's it. That's the core mechanism. No magic, just probability adjustment.

**Analogy:**
Think of the base model as someone who's read the entire internet but never focused on cookbooks. Fine-tuning is like having them read only cookbooks for a week. They don't gain taste buds or cooking skills—they just learn to write text that sounds like a cookbook.

---

## **Stage 2: Preparing the Recipe Dataset**

### **Learning Focus: Data is Always Haunted**
*Core Skills: Dataset loading, cleaning messy data, text formatting*

**Time Estimate:** 1-2 hours

**What You'll Build:** A clean, formatted dataset of recipes ready for training.

**Critical Lesson:** Real-world data is messy. Always. Cleaning it is half the job.

---

### **Step 1: Load Recipe Dataset**

```python
from datasets import load_dataset

# Load RecipeNLG dataset (large collection of recipes)
print("Loading dataset... (this may take a minute)")
# If "mbien/recipe_nlg" fails, try "recipe_nlg" as the dataset ID
dataset = load_dataset("mbien/recipe_nlg", split="train")

# Let's look at what we have
print(f"Total recipes: {len(dataset):,}")
print(f"\nSample recipe:")
print(dataset[0])
```

**What you should see:** A dictionary with fields like:
- `title`: Recipe name
- `ingredients`: List of ingredients (or sometimes a string!)
- `directions`: Cooking instructions (or sometimes a list!)

---

### **Step 2: Inspect the Mess**

Real data is never clean. Let's see what we're dealing with:

```python
# Look at a few examples to find issues
for i in range(5):
    print(f"\n--- Recipe {i} ---")
    print(f"Title: {dataset[i]['title']}")
    print(f"Ingredients type: {type(dataset[i]['ingredients'])}")
    print(f"Directions type: {type(dataset[i]['directions'])}")
    
    # Show first ingredient
    if isinstance(dataset[i]['ingredients'], list):
        print(f"First ingredient: {dataset[i]['ingredients'][0]}")
    else:
        print(f"Ingredients (string): {dataset[i]['ingredients'][:100]}")
```

**What you'll discover:**
- Sometimes `ingredients` is a list: `['flour', 'sugar', 'eggs']`
- Sometimes it's a string: `"flour, sugar, eggs"`
- Same with `directions`!
- Some titles have weird characters
- Some recipes are incomplete

**This is normal.** Data is always haunted by inconsistencies.

---

### **Step 3: Format Recipes (Handling the Mess)**

We need to convert recipes into a consistent text format the model can learn from.

```python
def format_recipe(example):
    """
    Convert a recipe dictionary into formatted training text.
    Handles both list and string formats for ingredients and directions.
    
    Format:
    RECIPE: [Title]
    INGREDIENTS: [ingredients list]
    DIRECTIONS: [directions]
    [END]
    """
    # Get title
    title = example['title']
    
    # Handle ingredients (might be string OR list!)
    if isinstance(example['ingredients'], list):
        ingredients = ', '.join(example['ingredients'])
    else:
        # Already a string
        ingredients = example['ingredients']
    
    # Handle directions (might be string OR list!)
    if isinstance(example['directions'], list):
        # Join list items with periods
        directions = ' '.join(example['directions'])
    else:
        # Already a string
        directions = example['directions']
    
    # Create formatted text
    formatted = f"""RECIPE: {title}
INGREDIENTS: {ingredients}
DIRECTIONS: {directions}
[END]"""
    
    return {"text": formatted}

# Test the formatter
print("FORMATTED EXAMPLE:")
print(format_recipe(dataset[0])['text'][:400])
```

**Why this format?**
- **Consistent structure** helps the model learn patterns
- **Clear markers** (RECIPE, INGREDIENTS, DIRECTIONS) give the model cues
- **[END] token** tells the model when to stop generating (we'll use this later)

---

### **Step 4: Filter Out Bad Data**

Not all recipes are worth training on. Let's filter for quality:

```python
def is_valid_recipe(example):
    """
    Filter out low-quality recipes.
    
    Criteria:
    - Must have ingredients
    - Must have directions
    - Must have at least 3 ingredients (not too simple)
    - Title shouldn't be too short
    """
    # Check ingredients exist and have substance
    ingredients = example['ingredients']
    if isinstance(ingredients, list):
        if len(ingredients) < 3:
            return False
    else:
        # If string, check it's not empty or too short
        if len(ingredients.strip()) < 20:
            return False
    
    # Check directions exist
    directions = example['directions']
    if isinstance(directions, list):
        if len(directions) < 1:
            return False
    else:
        if len(directions.strip()) < 30:
            return False
    
    # Check title is reasonable
    if len(example['title'].strip()) < 3:
        return False
    
    return True

# Filter the dataset
before_count = len(dataset)
dataset = dataset.filter(is_valid_recipe)
after_count = len(dataset)

print(f"Before filtering: {before_count:,} recipes")
print(f"After filtering: {after_count:,} recipes")
print(f"Removed: {before_count - after_count:,} low-quality recipes")
```

**Why filtering matters:**
- Garbage in = garbage out
- Training on incomplete recipes teaches the model bad patterns
- Quality over quantity

---

### **Step 5: Process and Split the Dataset**

```python
# We'll use a subset for faster training (full dataset would take hours)
# Select 5000 recipes for training, 500 for validation
small_dataset = dataset.select(range(5500))

# Apply formatting to all recipes
print("Formatting all recipes...")
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
- **Never test on training data**—that's like studying with the test answers!

**Why only 5,000 recipes?**
- Full dataset has 2 million+ recipes
- Would take 10+ hours on free Colab
- 5,000 is enough to see dramatic improvement

---

### **Checkpoint #2**

**Verify your data:**
```python
# Look at a formatted training example
print("Sample formatted recipe:")
print(train_dataset[0]['text'])
print("\n" + "="*60)
print("Character count:", len(train_dataset[0]['text']))
```

**You should see:** 
- A cleanly formatted recipe with clear structure
- RECIPE, INGREDIENTS, DIRECTIONS sections
- [END] marker at the end
- No Python list syntax (e.g., no brackets or quotes)

---

### **Common Issues & Debugging**

**Problem:** Some recipes still look weird
- **This is normal!** Some messiness always remains
- The model will learn to ignore occasional junk

**Problem:** Dataset filtering removed too many recipes
- **Solution:** Loosen the criteria in `is_valid_recipe()`

**Problem:** Formatting error on some recipes
- **Solution:** Add try/except in `format_recipe()` to skip problematic ones

---

## **Stage 3: Tokenizing the Data**

### **Learning Focus: How Models Actually Process Text**
*Core Skills: Tokenization, padding, sequence length*

**Time Estimate:** 45 minutes

**What You'll Build:** Convert text recipes into numerical tensors the model can train on.

---

### **Step 1: Configure the Tokenizer**

```python
# Set padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

# Add custom [END] token as a special token
tokenizer.add_special_tokens({"additional_special_tokens": ["[END]"]})
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    """
    Convert text to tokens (numbers) with fixed length.
    """
    # Tokenize with truncation and padding
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=256,  # Reduced from 512 to avoid memory issues
        padding='max_length'
    )

# Apply tokenization to entire dataset
print("Tokenizing training data...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

print("Tokenizing validation data...")
tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

print("✅ Data tokenized and ready for training")
```

**What's happening:**
- **Tokenization**: Text → numbers (e.g., "cookie" → 14523)
- **Truncation**: Cuts recipes longer than 256 tokens (about 200 words)
- **Padding**: Makes all sequences same length (needed for batch processing)

**Why max_length=256?**
- Shorter sequences = less GPU memory needed
- Most recipes fit in 256 tokens
- Prevents out-of-memory errors on free Colab

---

### **Step 2: Inspect Tokenized Data**

```python
# Let's see the transformation
sample_idx = 0

print("Original text:")
print(train_dataset[sample_idx]['text'][:200])
print("\nTokenized (first 30 tokens):")
print(tokenized_train[sample_idx]['input_ids'][:30])
print(f"\nTotal tokens in this recipe: {len(tokenized_train[sample_idx]['input_ids'])}")

# Decode back to text to verify
decoded = tokenizer.decode(tokenized_train[sample_idx]['input_ids'])
print("\nDecoded back to text (first 200 chars):")
print(decoded[:200])
```

**What to observe:**
- Text becomes a list of integers
- Some tokens are padding (will be ignored during training)
- Decoding works—we can go back and forth

---

### **Checkpoint #3**

**Verify tokenization worked:**
```python
# Check dataset structure
print(f"Training samples: {len(tokenized_train)}")
print(f"Validation samples: {len(tokenized_val)}")
print(f"Sample keys: {tokenized_train[0].keys()}")
print(f"Sequence length: {len(tokenized_train[0]['input_ids'])}")
```

**Expected output:**
- Should see 'input_ids' and 'attention_mask'
- All sequences should be length 256
- No errors

---

## **Stage 4A: Understanding One Training Step**

### **Learning Focus: What Training Actually Does**
*Core Skills: Forward pass, loss calculation, backpropagation*

**Time Estimate:** 45 minutes

**What You'll Learn:** Before we automate training, let's see what happens in ONE training step.

**The Big Idea:** Training is just:
1. Show model some text
2. Ask "how surprised are you?"
3. Adjust weights to be less surprised next time
4. Repeat millions of times

---

### **Step 1: Prepare One Batch**

```python
from transformers import DataCollatorForLanguageModeling

# Set up data collator (handles batching)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked LM
)

# Create a small dataloader with just one batch
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_train,
    batch_size=4,
    collate_fn=data_collator,
    shuffle=True
)

# Get one batch
batch = next(iter(train_dataloader))
print("Batch keys:", batch.keys())
print("Batch shape:", batch['input_ids'].shape)
print("Batch size: 4 recipes, each 256 tokens long")
```

---

### **Step 2: Run One Forward Pass**

```python
# Move batch to GPU
batch = {k: v.to(device) for k, v in batch.items()}

# Forward pass: model makes predictions
outputs = model(**batch)

# Extract loss
loss = outputs.loss

print(f"Loss (before training): {loss.item():.4f}")
print("\nWhat is loss?")
print("- Loss measures how SURPRISED the model is by the text")
print("- High loss = very surprised (bad predictions)")
print("- Low loss = not surprised (good predictions)")
print("- Goal: Make loss go down")
```

**Understanding loss:**
- Loss is the average "surprise" per token
- If the model predicts "sugar" and sees "sugar" → low surprise (small loss)
- If the model predicts "engine" and sees "sugar" → big surprise (large loss)
- Loss around 3-4 = model is guessing randomly
- Loss around 1-2 = model has learned patterns

---

### **Step 3: Backward Pass (Learning)**

```python
# Backward pass: calculate how to adjust weights
loss.backward()

print("✅ Gradients calculated!")
print("\nWhat just happened?")
print("- PyTorch calculated: 'How should I adjust each weight to reduce loss?'")
print("- This creates 'gradients' for all 82 million parameters")
print("- Next step would be to apply these gradients (update weights)")
print("\nWe won't actually update here—just demonstrating the mechanics")

# Clear gradients (cleanup)
model.zero_grad()
```

**The Training Loop:**
```
For each batch of recipes:
    1. Forward pass → get loss
    2. Backward pass → get gradients  
    3. Update weights → reduce loss
    4. Repeat
```

**Why we're not updating weights here:** 
We're just demonstrating. The Trainer (next stage) will do this automatically.

---

### **Checkpoint #4A**

**What you should understand:**
- ✅ Loss is a measure of surprise
- ✅ Forward pass produces predictions and loss
- ✅ Backward pass calculates gradients
- ✅ Training = repeating this process thousands of times

**If loss is around 3-4:** Normal for untrained model on new data

---

## **Stage 4B: Fine-Tuning with LoRA**

### **Learning Focus: Modern Parameter-Efficient Training**
*Core Skills: LoRA configuration, efficient fine-tuning, checkpointing*

**Time Estimate:** 2-3 hours (mostly waiting for training)

**What You'll Build:** Actually train the model using LoRA (Low-Rank Adaptation).

---

### **Conceptual Pause: Why LoRA Instead of Full Fine-Tuning?**

**Traditional Fine-Tuning:**
- Updates all 82 million parameters
- Requires lots of GPU memory
- Slow
- Easy to overfit

**LoRA (Low-Rank Adaptation):**
- Freezes the original 82 million parameters
- Adds small "adapter" layers (~500,000 new parameters)
- Only trains the adapters
- Much faster and more memory-efficient

**Analogy:**
- **Full fine-tuning**: Remodel the entire house
- **LoRA**: Add some furniture and decorations (keeps the structure, adapts the style)

**Modern Reality:**
- Most fine-tuning today uses LoRA or similar techniques
- It's faster, cheaper, and often works better
- This is how ChatGPT, Claude, and Gemini are adapted for specific tasks

---

### **Step 1: Configure LoRA**

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
# Fill in the target modules:
lora_config = LoraConfig(
    r=8,                          # Rank of adapter matrices (higher = more capacity)
    lora_alpha=16,                # Scaling factor
    target_modules=["???"],       # Which layers to adapt? (hint: attention)
    lora_dropout=0.05,            # Dropout for regularization
    bias="none",                  # Don't adapt bias terms
    task_type="CAUSAL_LM"         # Causal language modeling
)
<!-- PARTIAL_REVEAL -->
# Complete solution:
lora_config = LoraConfig(
    r=8,                          # Rank of adapter matrices
    lora_alpha=16,                # Scaling factor
    target_modules=["c_attn"],    # Adapt attention layers
    lora_dropout=0.05,            # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Understanding the parameters:**
- **r=8**: Rank of the adapter matrices (how expressive the adapters are)
- **lora_alpha=16**: Controls the magnitude of adapter updates
- **target_modules=["c_attn"]**: Which parts of the model to adapt (attention layers)
- **lora_dropout=0.05**: Prevents overfitting

---

### **Step 2: Apply LoRA to Model**

```python
# Apply LoRA configuration to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
```

**What you should see:**
```
trainable params: 294,912 || all params: 82,125,312 || trainable%: 0.36%
```

**This is the magic:**
- We're only training ~300,000 parameters (0.36%)
- The original 82 million are frozen
- This is why LoRA is so efficient!

---

### **Step 3: Set Up Training Configuration**

```python
from transformers import TrainingArguments, Trainer

# Configure training parameters
training_args = TrainingArguments(
    output_dir="./recipe-lora-model",
    num_train_epochs=3,                    # 3 passes through data
    per_device_train_batch_size=4,         # 4 recipes per batch
    per_device_eval_batch_size=4,
    save_steps=500,                        # Save checkpoint every 500 steps
    save_total_limit=2,                    # Keep only 2 checkpoints (saves space)
    logging_steps=100,                     # Log progress every 100 steps
    evaluation_strategy="steps",           # Evaluate periodically
    eval_steps=500,                        # Evaluate every 500 steps
    learning_rate=5e-5,                    # How fast to adjust weights
    warmup_steps=200,                      # Gradually increase learning rate
    weight_decay=0.01,                     # Regularization
    logging_dir='./logs',
    load_best_model_at_end=True,           # Load best checkpoint at end
    report_to="none"                       # Don't send logs to external services
)

print("✅ Training configuration set")
```

**Understanding training parameters:**
- **Epochs**: How many times to see the entire dataset (more ≠ always better)
- **Batch size**: Recipes processed simultaneously (limited by GPU memory)
- **Learning rate**: How much to adjust weights each step (5e-5 = 0.00005)
- **Warmup**: Gradually increases learning rate at start (prevents instability)

---

### **Step 4: Create the Trainer**

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
print(f"Training on {len(tokenized_train)} recipes")
print(f"Validating on {len(tokenized_val)} recipes")
```

---

### **Step 5: Start Training!**

```python
# This is where the magic happens
print("🚀 Starting training...")
print("This will take ~10-15 minutes on a T4 GPU")
print("-" * 60)

trainer.train()

print("\n✅ Training complete!")
```

**What you'll see during training:**
```
Step    Loss    Validation Loss
100     2.845   2.789
200     2.456   2.421
300     2.187   2.156
...
```

**Expected behavior:**
- **Initial loss**: ~3.5-4.0 (model is surprised by recipe text)
- **Final loss**: ~1.5-2.0 (model learned recipe patterns)
- **Loss should decrease** steadily
- If loss increases or flatlines, something's wrong

**Training metrics:**
- **Training loss**: How well model fits training data
- **Validation loss**: How well model generalizes to new data
- **Good sign**: Both decrease together
- **Bad sign**: Training loss drops but validation loss increases (overfitting)

---

### **Step 6: Save the Fine-Tuned Model**

```python
# Save LoRA adapters
model.save_pretrained("./recipe-gpt2-lora")
tokenizer.save_pretrained("./recipe-gpt2-lora")

print("✅ LoRA adapters saved to ./recipe-gpt2-lora")
print("\nWhat was saved:")
print("- LoRA adapter weights (~2 MB)")
print("- Tokenizer configuration")
print("- NOT the full model (we only save the adapters!)")
```

**Important:** 
- We only save the adapter weights, not the full model
- To use this later, load the base model + adapters
- This keeps file sizes small

---

### **Checkpoint #4B**

**Verification checklist:**
- ✅ Training completed without errors
- ✅ Loss decreased over time (from ~3.5 to ~1.5-2.0)
- ✅ No out-of-memory errors
- ✅ Model and adapters saved successfully
- ✅ Total training time was reasonable (~10-20 min)

---

### **Common Training Issues**

**Problem:** "CUDA out of memory"
- **Solution:** Reduce `per_device_train_batch_size` to 2 or even 1
- **Why it happens:** GPU ran out of memory trying to process the batch

**Problem:** Loss isn't decreasing
- **Solution:** Check learning rate (try 1e-5 to 1e-4), verify data is formatted correctly
- **Debug:** Print a few training examples to check formatting

**Problem:** Training is very slow (>30 min)
- **Solution:** Confirm GPU is enabled (should see "cuda" in device check)
- **Check:** Runtime → Change runtime type → T4 GPU selected

**Problem:** Loss drops then spikes
- **Normal:** Some volatility is expected, especially early
- **If severe:** Reduce learning rate

**Problem:** Validation loss increases while training loss decreases
- **This is overfitting:** Model memorizing training data
- **Solution:** Stop earlier (fewer epochs) or increase dropout

---

## **Stage 5: Testing and Evaluating the Fine-Tuned Model**

### **Learning Focus: Quantitative and Qualitative Evaluation**
*Core Skills: Model loading, perplexity, before/after comparison*

**Time Estimate:** 1-1.5 hours

**What You'll Build:** A comprehensive evaluation showing how fine-tuning changed behavior.

---

### **Step 1: Load the Fine-Tuned Model**

```python
from peft import PeftModel

# Load base model
base_for_comparison = GPT2LMHeadModel.from_pretrained("distilgpt2")
base_for_comparison = base_for_comparison.to(device)

# Load fine-tuned model (base + LoRA adapters)
finetuned_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
finetuned_model = PeftModel.from_pretrained(finetuned_model, "./recipe-gpt2-lora")
finetuned_model = finetuned_model.to(device)

print("✅ Both models loaded")
print("✅ Base model: Original DistilGPT2")
print("✅ Fine-tuned model: DistilGPT2 + Recipe LoRA adapters")
```

---

### **Step 2: Quantitative Evaluation - Perplexity**

Before we look at generated text, let's measure improvement numerically:

```python
import math

# Evaluate base model
print("Evaluating base model...")
eval_results_base = Trainer(
    model=base_for_comparison,
    args=training_args,  # Use same args for consistency
    data_collator=data_collator,
    eval_dataset=tokenized_val
).evaluate()

base_loss = eval_results_base['eval_loss']
base_perplexity = math.exp(base_loss)

# Evaluate fine-tuned model
print("Evaluating fine-tuned model...")
eval_results_ft = trainer.evaluate()
ft_loss = eval_results_ft['eval_loss']
ft_perplexity = math.exp(ft_loss)

print("\n" + "="*60)
print("QUANTITATIVE RESULTS")
print("="*60)
print(f"Base Model Loss:        {base_loss:.4f}")
print(f"Fine-Tuned Model Loss:  {ft_loss:.4f}")
print(f"Improvement:            {base_loss - ft_loss:.4f} ({'better' if ft_loss < base_loss else 'worse'})")
print()
print(f"Base Model Perplexity:       {base_perplexity:.2f}")
print(f"Fine-Tuned Model Perplexity: {ft_perplexity:.2f}")
print(f"Reduction:                   {base_perplexity - ft_perplexity:.2f} ({(1 - ft_perplexity/base_perplexity)*100:.1f}% better)")
```

**Understanding Perplexity:**
- **Perplexity = exp(loss)** 
- Measures "how confused the model is"
- Lower = better
- **Note:** Perplexity is useful for before/after comparisons on the same dataset, not as a universal quality number.
- Perplexity of 50 means the model is as confused as if choosing randomly from 50 options
- **Expected results:**
  - Base model: perplexity ~30-50 (very confused about recipes)
  - Fine-tuned: perplexity ~5-10 (much less confused)

---

### **Step 3: Qualitative Evaluation - Generate and Compare**

Now let's see the actual text:

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
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("[END]")  # Stop at [END]
        )
    ft_text = tokenizer.decode(ft_output[0], skip_special_tokens=True)
    print(ft_text)
    print("\n" + "=" * 70 + "\n")
```

**Note the `eos_token_id`:**
- We tell the model to stop when it generates "[END]"
- This prevents infinite rambling
- The model learned this pattern from the training data

---

### **Step 4: Test with Various Prompts**

```python
# Test prompts
test_prompts = [
    "RECIPE: Chocolate Chip Cookies\nINGREDIENTS:",
    "RECIPE: Spaghetti Carbonara\nINGREDIENTS:",
    "RECIPE: Homemade Pizza\nINGREDIENTS:",
    "RECIPE: Chicken Stir Fry\nINGREDIENTS:",
    "RECIPE: Apple Pie\nINGREDIENTS:",
]

# Run comparisons
for prompt in test_prompts:
    compare_models(prompt, base_for_comparison, finetuned_model, max_length=250)
```

---

### **Checkpoint #5**

**What you should observe:**

**Base Model (🔵):** 
- Inconsistent format
- Goes off-topic quickly
- Vague or nonsensical ingredients
- Doesn't follow recipe structure
- May talk about unrelated things

**Fine-Tuned Model (🟢):**
- Follows RECIPE/INGREDIENTS/DIRECTIONS format
- Lists plausible ingredients
- Generates coherent cooking steps
- Stays on-topic
- Stops at [END]

**Example comparison:**

```
PROMPT: RECIPE: Chocolate Chip Cookies
INGREDIENTS:

🔵 BASE MODEL:
2 cups of flour, 1 tablespoon of baking powder, and some other stuff I found in the pantry. 
You can also add whatever you want. The recipe is pretty flexible and you could probably...

🟢 FINE-TUNED MODEL:
2 cups all-purpose flour, 1 teaspoon baking soda, 1 teaspoon salt, 
1 cup butter softened, 3/4 cup granulated sugar, 3/4 cup packed brown sugar, 
2 eggs, 2 teaspoons vanilla extract, 2 cups chocolate chips
DIRECTIONS: Preheat oven to 375°F. In a bowl, mix flour, baking soda, and salt...
[END]
```

**The difference is dramatic!**

---

### **Step 5: Analysis Questions**

Create a markdown cell and answer:

**1. What changed?** 
Describe 3 specific differences between base and fine-tuned outputs.

**2. Why does fine-tuning work?** 
In your own words, explain why the model got better at recipes. Use the concept of "probability distributions."

**3. What didn't change?** 
The model still has the same architecture and 82 million base parameters—what did training actually modify?

**4. Is the model "intelligent" about cooking?**
Does it understand taste? Nutrition? Chemistry? What does it actually "know"?

---

## **Stage 6: Experimenting with Generation Parameters**

### **Learning Focus: Controlling Model Behavior**
*Core Skills: Hyperparameter tuning, temperature, top-p sampling*

**Time Estimate:** 45 minutes

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
                max_length=200,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("[END]")
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

**Questions to answer:**
1. Which temperature gives the most realistic recipe?
2. Which gives the most creative (even if weird)?
3. When might you want high temperature? Low temperature?

---

### **Step 2: Top-P (Nucleus) Sampling**

```python
def test_top_p(prompt, model, top_p_values=[0.5, 0.9, 0.95, 1.0]):
    """
    Generate text with different top-p values.
    """
    print(f"PROMPT: {prompt}\n")
    
    for top_p in top_p_values:
        print(f"☁️ Top-P = {top_p}")
        print("-" * 60)
        
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=200,
                temperature=0.8,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("[END]")
            )
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(text)
        print("\n")

# Test it
test_top_p(
    "RECIPE: Vegetable Soup\nINGREDIENTS:",
    finetuned_model
)
```

**Understanding top-p:**
- Controls the cumulative probability mass to sample from
- 0.9 = sample from smallest set of tokens that sum to 90% probability
- Lower = more focused, higher = more diverse
- Works with temperature to control randomness

---

### **Step 3: Create Your Own Recipe Generator**

```python
def generate_custom_recipe(dish_name, temperature=0.8, top_p=0.9):
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
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("[END]")
        )
    
    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return recipe

# Try your own recipes!
my_dishes = [
    "Garlic Butter Shrimp",
    "Vegetarian Tacos",
    "Lemon Cake",
    "Thai Green Curry",
    "Homemade Pasta"
]

for dish in my_dishes:
    print("=" * 70)
    print(generate_custom_recipe(dish))
    print("\n")
```

---

### **Checkpoint #6**

**Experiments to complete:**
- ✅ Tested at least 3 different temperatures
- ✅ Tested at least 3 different top-p values
- ✅ Generated at least 5 custom recipes
- ✅ Documented which settings work best for your use case

**Analysis:**
Write a markdown cell explaining:
1. What's the sweet spot for temperature? Why?
2. How does top-p affect output quality?
3. If you were building a recipe app, which settings would you use?

---

## **Project Deliverables**

Submit your Colab notebook with:

### **1. Code (Well-Commented)**
- All stages completed and working
- Clear comments explaining each section
- Organized with markdown headers
- No errors when run top-to-bottom

### **2. Outputs**
Include:
- Base model generation examples (Stage 1)
- Formatted and cleaned dataset samples (Stage 2)
- Tokenization examples (Stage 3)
- Training loss curves or logs (Stage 4)
- Perplexity comparison (Stage 5)
- Side-by-side base vs fine-tuned comparisons (Stage 5)
- Temperature experiment results (Stage 6)

### **3. Written Responses**

Answer these questions in markdown cells:

**Understanding Questions:**
1. What is fine-tuning? Explain in 2-3 sentences to someone who doesn't know ML.
2. What actually changes inside the model during fine-tuning? (Hint: probability distributions)
3. Why did we use LoRA instead of full fine-tuning?
4. Why did we use only 5,000 recipes instead of the full dataset (2 million+)?
5. What would happen if we trained for 20 epochs instead of 3?

**Analysis Questions:**
6. Compare the base vs fine-tuned model:
   - Which follows recipe structure better?
   - Which stays on-topic more consistently?
   - Which would you trust for actual cooking?
   - What specific patterns did the model learn?

7. Based on your temperature experiments:
   - What temperature gave the best results?
   - When might you want high temperature? Low temperature?
   - What's the tradeoff between creativity and coherence?

8. Perplexity analysis:
   - What was the perplexity before and after fine-tuning?
   - What does this number actually measure?
   - Why is lower perplexity better?

**Application Questions:**
9. Name three other domains where fine-tuning could be useful. For each, describe:
   - What dataset you'd need
   - What behavior you're trying to teach
   - Example: "Medical diagnosis: dataset of symptom→diagnosis pairs, teaching medical reasoning patterns"

10. If you wanted to fine-tune a model to write Python code, what would you need?
    - What dataset?
    - What format?
    - What would be different from recipes?

**Critical Thinking:**
11. Limitations: What can this model NOT do?
    - Can it taste food?
    - Can it understand nutrition?
    - Can it reason about cooking chemistry?
    - What is it actually doing?

12. Data quality: How did messy data affect the model?
    - What problems did you encounter?
    - How did you solve them?
    - What would happen if you didn't clean the data?

**Reflection:**
13. What surprised you most about this project?
14. What was challenging about preparing the dataset?
15. How is fine-tuning different from "training a model from scratch"?
16. What would you do differently if you did this again?

---

## **Success Criteria**

### **Complete (Meets Expectations)**
- ✅ Stages 1-5 fully functional
- ✅ Base model output captured
- ✅ Data cleaning implemented
- ✅ LoRA fine-tuning completed successfully
- ✅ Perplexity calculated and compared
- ✅ Clear before/after comparison
- ✅ All analysis questions answered
- ✅ Code runs without errors

### **Proficient (Exceeds Expectations)**
- ✅ All of above, plus:
- ✅ Temperature experiments completed (Stage 6)
- ✅ Code is well-documented with explanations
- ✅ Thoughtful analysis of results showing deep understanding
- ✅ Tested with creative/unusual prompts
- ✅ Discussed limitations and failure cases
- ✅ Clear explanations of concepts (perplexity, LoRA, etc.)

### **Excellent (Outstanding)**
- ✅ All of above, plus:
- ✅ Trained for different epoch counts and compared results
- ✅ Experimented with different LoRA configurations (rank, alpha)
- ✅ Tested different dataset sizes (1K, 5K, 10K)
- ✅ Analyzed failure cases in detail
- ✅ Created visualizations (loss curves, perplexity plots)
- ✅ Clear documentation that could teach others
- ✅ Demonstrated understanding of when/why to use fine-tuning

### **Exceptional (Goes Above and Beyond)**
- ✅ All of above, plus:
- ✅ Fine-tuned on a different dataset (your choice!)
- ✅ Compared LoRA to full fine-tuning
- ✅ Built evaluation metrics beyond perplexity
- ✅ Created a simple interface (Gradio/Streamlit)
- ✅ Implemented custom stopping criteria or generation tricks
- ✅ Explored advanced topics (LoRA rank sensitivity, learning rate schedules)
- ✅ Could serve as a teaching resource for future students

---

## **Tips for Success**

### **Before You Start**
- Read the entire assignment to understand the flow
- Ensure GPU is enabled (Runtime → Change runtime type → T4 GPU)
- Budget 30 minutes for training—don't start right before you need to leave
- Create a plan: which stages will you complete in each session?

### **While Working**
- Save frequently (File → Save or Ctrl+S)
- Run cells in order—don't skip ahead
- If training fails, check GPU memory and reduce batch size
- Keep a notes section for observations
- Document interesting findings as you go
- Test on simple examples before complex ones

### **When You Get Stuck**
1. **Check error messages** carefully—they usually tell you what's wrong
2. **Verify GPU** is enabled and being used
3. **Restart runtime** if things get corrupted (Runtime → Restart runtime)
4. **Reduce batch size** if you see CUDA memory errors
5. **Check your data** if loss isn't decreasing—print some examples
6. **Ask for help** with specific error messages and what you've tried

### **Best Practices**
- Comment your code as you write
- Use markdown cells to explain your thinking
- Save before starting training (sessions can crash)
- Keep all your experiments in the notebook
- Document both successes and failures
- Take screenshots of interesting results

### **Debugging Checklist**
If something's not working:
- [ ] Is GPU enabled?
- [ ] Did I run cells in order?
- [ ] Are my dataset samples properly formatted?
- [ ] Did training actually complete?
- [ ] Are loss values reasonable (not NaN)?
- [ ] Did I save the model correctly?
- [ ] Am I loading the right model for comparison?

---

## **Extensions & Next Steps**

After completing this project, you could:

### **Easy Extensions:**
1. **Different Datasets:** Fine-tune on song lyrics, movie scripts, or academic papers
2. **Longer Sequences:** Try max_length=512 (if GPU allows)
3. **More Epochs:** Train for 5-10 epochs and track performance
4. **Different Models:** Try GPT-2 (124M params) or GPT-2-medium (355M)

### **Moderate Extensions:**
5. **Instruction Following:** Fine-tune on instruction-following datasets
6. **Evaluation Metrics:** Implement BLEU scores or ROUGE metrics
7. **Gradio Interface:** Create a web UI for recipe generation
8. **Multi-Task:** Combine recipes + other domains in one model
9. **Different LoRA Configs:** Test r=16, r=32, different target modules

### **Advanced Extensions:**
10. **Custom Dataset Collection:** Scrape recipes from websites
11. **Active Learning:** Have users rate outputs, retrain on best ones
12. **Reinforcement Learning:** Use RLHF to improve quality
13. **Vision Integration:** Fine-tune a multimodal model on recipe images
14. **Deployment:** Deploy to HuggingFace Spaces or as an API

**Want to go deeper?**
- Learn about QLoRA (quantized LoRA for even larger models)
- Explore RLHF (reinforcement learning from human feedback)
- Study the HuggingFace course on fine-tuning
- Try fine-tuning vision models or audio models
- Read papers on parameter-efficient fine-tuning

---

## **Stage 7 (Exceptional): Build Your Own Dataset**

### **Learning Focus: End-to-End ML Pipeline**
*Core Skills: Data collection, cleaning, formatting, evaluation*

**Time Estimate:** 3-5 hours

**What You'll Build:** Collect your own dataset, fine-tune on it, and compare to the RecipeNLG model.

---

### **Assignment:**

1. **Choose a Domain:**
   - Song lyrics from a specific artist
   - Movie dialogue from a specific genre
   - Product descriptions from a specific category
   - Social media posts from a specific topic
   - Academic abstracts from a specific field

2. **Collect Data:**
   - Gather at least 500 examples
   - Save as a CSV or JSON file
   - Document your sources

3. **Clean and Format:**
   - Remove duplicates
   - Fix encoding issues
   - Create consistent format
   - Filter low-quality examples

4. **Fine-Tune:**
   - Use the same LoRA approach
   - Train for 3 epochs
   - Save checkpoints

5. **Evaluate:**
   - Compare to base model
   - Calculate perplexity
   - Generate examples
   - Compare to RecipeNLG-trained model

6. **Document:**
   - Describe your data collection process
   - What challenges did you face?
   - How does this dataset compare to RecipeNLG?
   - What did you learn?

---

### **Deliverables:**
- Dataset file (CSV/JSON)
- Data collection script (if automated)
- Cleaning/formatting notebook
- Fine-tuning notebook
- Comparison analysis
- Reflection on the process

**This is the real deal:** You've now done what professional ML engineers do every day!

---

## **Key Takeaways**

### **What is fine-tuning?**
- Taking a pre-trained model and specializing it for a specific task
- Adjusts weights slightly to fit new data patterns
- Much faster and cheaper than training from scratch
- Uses techniques like LoRA to be efficient

### **Why does it matter?**
- How ChatGPT became helpful (started as base GPT, fine-tuned on instructions)
- How medical AI, legal AI, coding assistants are built
- Core technique in modern AI development
- Enables specialized AI without massive compute budgets

### **What did you learn?**
- **Models are adaptable** through additional training
- **Data quality matters** enormously (garbage in = garbage out)
- **Data is always haunted** by inconsistencies and errors
- **Trade-offs exist** (epochs, batch size, temperature, LoRA rank)
- **Even small models** can be specialized effectively
- **The model doesn't "understand"**—it's learning probability distributions

### **What actually changed?**
- **NOT**: The model's architecture or number of parameters
- **NOT**: The model's "intelligence" or "understanding"
- **YES**: The probability distributions over tokens
- **YES**: Small adapter weights that nudge predictions
- **RESULT**: Recipe-shaped text becomes more likely

### **Critical Understanding:**
The model didn't learn what food tastes like, or cooking chemistry, or nutrition. It learned:
> "After the token 'INGREDIENTS:' usually comes food-related nouns, followed by measurements, followed by 'DIRECTIONS:', followed by cooking verbs."

That's probability, not understanding. This is behavior shaping, not intelligence.

**You've experienced the foundation of how modern AI systems are created! 🚀**

---

## **Final Reflection Questions**

1. **Before this project, what did you think fine-tuning was?**

2. **What's the most important thing you learned?**

3. **How has your understanding of AI changed?**

4. **What do you want to explore next?**

5. **How would you explain fine-tuning to a friend who's never coded?**

---

**Congratulations on completing this project!** You've learned one of the most important techniques in modern machine learning. The skills you've developed here—data preparation, training, evaluation, and critical thinking about AI—will serve you well in any ML project.

The universe rewards those who understand what's actually happening, not just those who press buttons. You've earned your place in the first category. ✨
