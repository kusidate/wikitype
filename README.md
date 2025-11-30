# 📝 "wikitype": Typing Practice Program User Manual

## Overview

This program is a console-based typing practice for Japanese and English application written in Rust. It uses Wikipedia articles, local text files, or a list of past mistakes as source material to measure and evaluate the user's typing speed (WPM/KPM) and accuracy.  
Wikipedia texts are licenced under CC BY-SA 3.0.

## System Requirements

  * **Language:** Rust
  * **Asynchronous Runtime:** `tokio`
  * **Main Dependencies:** `anyhow`, `kana-converter`, `lindera` (Japanese morphological analysis), `reqwest`, `serde`, `rand`, `async-trait`

## How to Run

The program is started with `tokio::main`. After building the executable, run it from the terminal:

```bash
./wikitype # Example: the name of the executable after building
```

## Main Menu and Source Selection

When the program starts, the following options are displayed:

| Number | Source | Language | Description |
| :--- | :--- | :--- | :--- |
| **1** | Wikipedia | Japanese (ja) | Fetches a random Japanese Wikipedia article and uses it as a challenge, line by line. |
| **2** | Wikipedia | English (en) | Fetches a random English Wikipedia article and uses it as a challenge, line by line. |
| **3** | Mistakes List | Japanese (ja) | Selects tokens (words) that the user has previously made mistakes on in Japanese for the challenge. |
| **4** | Mistakes List | English (en) | Selects tokens (words) that the user has previously made mistakes on in English for the challenge. |
| **5〜** | Local File | (File Dependent) | Uses `.txt` files located in the same directory as the program for the challenge. |

> **💡 Tip:** If you press the Enter key without typing anything, or enter an invalid number after a round, the program will **refresh and reuse the previous source**.

### Local File Format

When using a local file (e.g., `sample.txt`), the file must adhere to the following structure:

1.  **Line 1 (Header):** `language_code,token_count,display_title`
      * `language_code`: Must be `ja` or `en`.
      * `token_count`: The maximum number of random tokens to be selected (must be 5 or more).
      * `display_title`: The title to be shown in the menu.
2.  **Line 2 onwards:** Lines of text to be used as challenges.
      * If a line starts with a **hyphen (`-`)**, the entire line is used as a single sentence challenge (Full Sentence Challenge).
      * If there is no hyphen, the line is treated as a token list, and the specified **number of tokens are randomly selected**, combined, and presented as the challenge (Random Token Challenge).

**Example (`japanese_tokens.txt`):**

```text
ja,10,Common Vocabulary
-This is a full sentence challenge.
typing practice development project Rust language
program creation editor file terminal
```

## Typing Session Flow

1.  After selecting the source, the session starts with the message: "`=== Starting [Source Title] Mode ===`".
2.  The challenge text is fetched and displayed.
      * For Japanese, a reading hint, such as Romaji, is displayed below the text under "`--- Reading ---`" (using `lindera`'s dictionary information).
3.  The challenge text is shown under "`--- Input ---`". Start typing when the "`↓`" appears.
4.  Press the **Enter key** when you finish typing.
5.  The program evaluates your input and reports the results.
6.  The separator "`------------------------`" is displayed, and the program is ready for the next round (either returning to the main menu or proceeding with the `last_source` if set).

## Evaluation and Reports

### 1\. Difference Display (Token-Unit)

Upon completion of the input, a colored comparison result is displayed based on **token units (word units)**. This uses a **Minimum Edit Distance Algorithm (Dynamic Programming)** to align the Expected and Input tokens.

  * **`Green`**: Matched tokens.
  * **`Bold Red`**: Mistakes (Substitution, Deletion, or Insertion).
  * **`Gray`**: The **remainder** of the expected sentence that was not typed (the part judged as untyped by the alignment).
      * A maximum of 10 gray tokens are displayed, and any further tokens are abbreviated with `...`.

### 2\. Individual Report

Individual round results are reported.

  * **Accuracy:** Displays the character-level accuracy as a percentage.
      * This is calculated based on the **number of character mistakes in the portion you typed**, subtracting the minimum penalty for untyped characters from the full edit distance.
  * **Mistakes:** Reports the number of token-level errors (Substitution, Deletion, and Insertion).

### 3\. Overall Result

Typing speed is reported.

  * **WPM (Words Per Minute):** The number of words per minute (conventionally calculated as the number of characters / 5).
  * **KPM (Keystrokes Per Minute):** The number of keystrokes per minute (number of characters).

## Mistake List Management

### Adding to the Mistake List

  * Tokens identified as **Substitution** or **Deletion** in the token-unit difference display are automatically added to the `mistakes.json` file as **Mistake Items** (if they are not already present).
  * The `MistakeItem` structure includes `text`, `lang`, and `success_count`.

### Mistakes List Mode (`3` or `4`) Operation

1.  Only items with a `success_count` of **less than 5** are selected for practice.
2.  A maximum of 20 tokens are randomly selected, joined by spaces, and presented as the challenge.
3.  If this challenge is completed with **zero token mistakes (Perfect)**, the `success_count` for **all items** used in the round is incremented by 1.
4.  Items that reach a `success_count` of 5 are removed from the active mistake list.

5.  Mistake data is saved to `mistakes.json`.
