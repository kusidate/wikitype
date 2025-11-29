use anyhow::Result;
use kana_converter::{to_double_byte, KanaAndAscii};
use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::time::Instant;
use rand::seq::SliceRandom;
use rand::thread_rng;
use async_trait::async_trait;
use std::any::Any; // 追加: TextSourceをダウンキャストするために必要

// --- Macros and Utility Functions ---

macro_rules! color {
    ($text:expr, "red") => { format!("\x1b[31m{}\x1b[0m", $text) };
    ($text:expr, "green") => { format!("\x1b[32m{}\x1b[0m", $text) };
    ($text:expr, "bold_red") => { format!("\x1b[1m\x1b[31m{}\x1b[0m", $text) };
    ($text:expr, "gray") => { format!("\x1b[90m{}\x1b[0m", $text) }; 
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MistakeItem {
    text: String,
    lang: String,
    success_count: usize,
}

fn load_mistakes() -> Vec<MistakeItem> {
    if let Ok(data) = fs::read_to_string("mistakes.json") {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        Vec::new()
    }
}

fn save_mistakes(items: &Vec<MistakeItem>) {
    if let Ok(json) = serde_json::to_string_pretty(items) {
        let _ = fs::write("mistakes.json", json);
    }
}

#[derive(Deserialize)]
struct WikiSummary {
    extract: String,
}

async fn fetch_text(api_url: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(api_url)
        .header("User-Agent", "MyTypingApp/1.0 (https://example.com)")
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("HTTP error: {}", resp.status()));
    }

    let summary: WikiSummary = resp.json().await?;
    Ok(summary.extract)
}

/// Compares expected and input tokens to check if at least one token matches
fn has_matched_token(processor: &Box<dyn LanguageProcessor>, expected: &str, input: &str) -> bool {
    let expected_tokens = processor.tokenize(expected);
    let input_tokens = processor.tokenize(input);

    if input_tokens.is_empty() {
        return false;
    }

    for et in expected_tokens.iter() {
        if input_tokens.contains(et) {
            return true;
        }
    }
    false
}

// --- Language Processing Trait and Implementations ---

trait LanguageProcessor: Send + Sync {
    fn api_url(&self) -> &'static str;
    fn normalize(&self, text: &str) -> String;
    fn tokenize(&self, text: &str) -> Vec<String>;
    fn display_hint(&self, text: &str) -> Result<()>;
    fn should_skip(&self, text: &str) -> bool;
}

struct JapaneseProcessor {
    tokenizer: Tokenizer,
}

impl JapaneseProcessor {
    fn new() -> Result<Self> {
        let dictionary = load_dictionary("embedded://unidic")?;
        let segmenter = Segmenter::new(Mode::Normal, dictionary, None);
        Ok(Self {
            tokenizer: Tokenizer::new(segmenter),
        })
    }
}

impl LanguageProcessor for JapaneseProcessor {
    fn api_url(&self) -> &'static str {
        "https://ja.wikipedia.org/api/rest_v1/page/random/summary"
    }
    fn normalize(&self, text: &str) -> String {
        let no_space = text.replace('\u{3000}', "").replace(' ', "").replace('\n', "");
        to_double_byte(&no_space, KanaAndAscii).trim().to_string()
    }
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenizer
            .tokenize(text)
            .map_or_else(
                |_| Vec::new(),
                |tokens| tokens.into_iter().map(|t| t.surface.to_string()).collect(),
            )
    }
    fn display_hint(&self, text: &str) -> Result<()> {
        let mut tokens = self.tokenizer.tokenize(text)?;
        if tokens.is_empty() {
            return Ok(());
        }
        let reading: String = tokens
            .iter_mut()
            .map(|t| {
                let details = t.details();
                match details.get(6) {
                    Some(s) => s.to_string(),
                    None => t.surface.to_string(),
                }
            })
            .collect::<Vec<String>>()
            .join(" ");
        println!("--- Reading ---\n{}", reading);
        Ok(())
    }
    fn should_skip(&self, _text: &str) -> bool {
        false
    }
}

struct EnglishProcessor;
impl LanguageProcessor for EnglishProcessor {
    fn api_url(&self) -> &'static str {
        "https://en.wikipedia.org/api/rest_v1/page/random/summary"
    }
    fn normalize(&self, text: &str) -> String {
        text.chars()
            .map(|c| match c {
                '“' | '”' | '‘' | '’' => '\'',
                '—' | '–' | '―' => '-',
                '…' => '.',
                '[' | ']' | '{' | '}' | '#' | '\n' | '<' | '>' | '®' | '™' | '©' => ' ',
                _ => c,
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_string()).collect()
    }
    fn display_hint(&self, _text: &str) -> Result<()> {
        Ok(())
    }
    fn should_skip(&self, text: &str) -> bool {
        text.chars().any(|c| !c.is_ascii())
    }
}

// --- Data Source Abstraction (TextSource Trait) ---

// Needs clone to be passed to GameSession::source
#[derive(Clone)]
enum SourceMode {
    Wikipedia(WikipediaSource),
    LocalFile(LocalFileSource),
    MistakeList(MistakeListSource),
}

// Implements language and display_title on SourceMode, delegating to internal variants
impl SourceMode {
    pub fn language(&self) -> &str {
        match self {
            SourceMode::Wikipedia(s) => s.language(),
            SourceMode::LocalFile(s) => s.language(),
            SourceMode::MistakeList(s) => s.language(),
        }
    }

    pub fn display_title(&self) -> String {
        match self {
            SourceMode::Wikipedia(s) => s.display_title(),
            SourceMode::LocalFile(s) => s.display_title(),
            SourceMode::MistakeList(s) => s.display_title(),
        }
    }
    
    // MistakeListSource の場合は、ファイルから再ロードして最新の状態を返す
    pub fn refresh(self) -> Self {
        match self {
            SourceMode::MistakeList(s) => SourceMode::MistakeList(MistakeListSource::new(s.lang)),
            _ => self,
        }
    }
}

// Warning suppression: for methods defined in the trait but only used via trait objects
#[allow(dead_code)] 
#[async_trait]
trait TextSource: Send + Sync {
    fn language(&self) -> &str;
    async fn get_next_target(&mut self, processor: &dyn LanguageProcessor) -> Result<Option<String>>;
    
    // MistakeList mode uses GameSession::finalize_mistake_list instead of this for real finalization.
    fn finalize_token(&mut self, token: &str); 

    fn display_title(&self) -> String;
    fn clone_source(&self) -> SourceMode;

    // ADDED: For downcasting in GameSession
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// --- TextSource Implementation: WikipediaSource ---

#[derive(Clone)]
struct WikipediaSource {
    lang: String,
    // Holds the entire article; returns the next line each time run_round is called
    current_lines: Vec<String>,
    line_index: usize,
}

impl WikipediaSource {
    fn new(lang: String) -> Self {
        Self {
            lang,
            current_lines: Vec::new(),
            line_index: 0,
        }
    }
}

#[async_trait]
impl TextSource for WikipediaSource {
    fn language(&self) -> &str {
        &self.lang
    }
    fn display_title(&self) -> String {
        format!("Wikipedia ({})", &self.lang)
    }
    fn clone_source(&self) -> SourceMode {
        SourceMode::Wikipedia(self.clone())
    }

    async fn get_next_target(&mut self, processor: &dyn LanguageProcessor) -> Result<Option<String>> {
        if self.line_index >= self.current_lines.len() {
            println!(">>> Fetching Wikipedia article ({}) ...", self.lang);
            let mut retries = 3;
            self.current_lines.clear();
            self.line_index = 0;

            while retries > 0 {
                match fetch_text(processor.api_url()).await {
                    Ok(t) => {
                        let lines: Vec<String> = t.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
                        if !lines.is_empty() {
                            self.current_lines = lines;
                            break;
                        } else {
                            println!("Article is empty. Retrying fetch...");
                        }
                    }
                    Err(e) => {
                        eprintln!("Fetch failed: {} Retrying...", e);
                    }
                }
                retries -= 1;
                if retries == 0 {
                    eprintln!("Failed to fetch Wikipedia article. Aborting.");
                    return Ok(None);
                }
            }
        }
        
        while self.line_index < self.current_lines.len() {
            let line = self.current_lines[self.line_index].clone();
            self.line_index += 1;

            if processor.should_skip(&line) {
                continue;
            }
            let normalized = processor.normalize(&line);
            if normalized.is_empty() {
                println!("normalized_target is empty. Skipping to the next line in the article...");
                continue;
            }
            return Ok(Some(normalized));
        }

        Ok(None) // Used all lines
    }

    fn finalize_token(&mut self, _token: &str) {} // No-op
    fn as_any(&self) -> &dyn Any { self } // ADDED
    fn as_any_mut(&mut self) -> &mut dyn Any { self } // ADDED
}

// --- TextSource Implementation: LocalFileSource ---

#[derive(Clone)]
struct LocalFileSource {
    lang: String,
    title: String,
    token_count: usize,
    all_lines: Vec<String>,
}

#[async_trait]
impl TextSource for LocalFileSource {
    fn language(&self) -> &str {
        &self.lang
    }
    fn display_title(&self) -> String {
        format!("Local File [{}]", self.title)
    }
    fn clone_source(&self) -> SourceMode {
        SourceMode::LocalFile(self.clone())
    }

    async fn get_next_target(&mut self, processor: &dyn LanguageProcessor) -> Result<Option<String>> {
        println!(">>> Generating challenge from local file [{}]", self.title);
        let mut rng = thread_rng();
        let mut selected_line = self.all_lines.choose(&mut rng).map(|s| s.clone()).unwrap_or_default();
        if selected_line.is_empty() { return Ok(None); }

        let normalized_target: String;

        if selected_line.starts_with('-') {
            // Full sentence challenge
            selected_line = selected_line.trim_start_matches('-').to_string();
            normalized_target = processor.normalize(&selected_line);
        } else {
            // Random token challenge
            let tokens: Vec<&str> = selected_line.split_whitespace().collect();
            let mut chosen_tokens: Vec<&str> = Vec::new();
            for _ in 0..self.token_count {
                if let Some(token) = tokens.choose(&mut rng) {
                    chosen_tokens.push(*token);
                }
            }
            let combined_text = chosen_tokens.join(" ");
            normalized_target = processor.normalize(&combined_text);
        }
        
        if normalized_target.is_empty() {
            return Ok(None);
        }
        return Ok(Some(normalized_target));
    }

    fn finalize_token(&mut self, _token: &str) {} // No-op
    fn as_any(&self) -> &dyn Any { self } // ADDED
    fn as_any_mut(&mut self) -> &mut dyn Any { self } // ADDED
}

// --- TextSource Implementation: MistakeListSource ---

#[derive(Clone)]
struct MistakeListSource {
    lang: String,
    items: Vec<MistakeItem>,
    current_round_items: Vec<MistakeItem>, // ADDED: Current tokens being challenged
}

impl MistakeListSource {
    fn new(lang: String) -> Self {
        // MistakeList のファイルからロード
        let items = load_mistakes().into_iter().filter(|i| i.lang == lang).collect();
        Self { 
            lang, 
            items, 
            current_round_items: Vec::new(), // Initialize
        }
    }
    fn save_current_mistakes(&self) {
        let mut all_mistakes = load_mistakes().into_iter().filter(|i| i.lang != self.lang).collect::<Vec<_>>();
        all_mistakes.extend_from_slice(&self.items);
        save_mistakes(&all_mistakes);
    }
}

#[async_trait]
impl TextSource for MistakeListSource {
    fn language(&self) -> &str {
        &self.lang
    }
    fn display_title(&self) -> String {
        if self.lang == "ja" { "Japanese Mistakes List".to_string() } else { "English Mistakes List".to_string() }
    }
    fn clone_source(&self) -> SourceMode {
        SourceMode::MistakeList(self.clone())
    }

    async fn get_next_target(&mut self, _processor: &dyn LanguageProcessor) -> Result<Option<String>> {
        let available: Vec<&MistakeItem> = self.items.iter().filter(|i| i.success_count < 5).collect();
        
        if available.is_empty() { 
            println!("Mistakes list is empty."); 
            self.current_round_items.clear(); // Clear to ensure no lingering state
            return Ok(None); 
        }

        let mut rng = thread_rng();
        let max_count = 20;
        let mut selected: Vec<MistakeItem> = Vec::new();

        for _ in 0..max_count {
            if let Some(item) = available.choose(&mut rng) {
                let count_in_selected = selected.iter().filter(|i| i.text == item.text).count();
                if count_in_selected < 5 {
                    selected.push((*item).clone());
                }
            }
        }

        if selected.is_empty() { 
            println!("No tokens available to challenge."); 
            self.current_round_items.clear();
            return Ok(None); 
        }

        // Store selected items for later finalization
        self.current_round_items = selected; 

        let combined_text = self.current_round_items.iter().map(|i| i.text.clone()).collect::<Vec<_>>().join(" ");
        
        // In mistake list mode, we return the combined tokens directly as the target.
        Ok(Some(combined_text))
    }

    // This method is no longer used for the actual finalization logic in MistakeList mode
    fn finalize_token(&mut self, _token: &str) {
        // No-op
    }

    fn as_any(&self) -> &dyn Any { self } // ADDED
    fn as_any_mut(&mut self) -> &mut dyn Any { self } // ADDED
}

// --- Game Session ---

struct GameSession {
    processor: Box<dyn LanguageProcessor>,
    source: Box<dyn TextSource>, 
    mistakes: Vec<MistakeItem>,
}

impl GameSession {
    fn new(source: SourceMode) -> Result<Self> {
        let lang = source.language().to_string(); 
        let processor: Box<dyn LanguageProcessor> = if lang == "ja" {
            Box::new(JapaneseProcessor::new()?)
        } else {
            Box::new(EnglishProcessor)
        };
        let source_box: Box<dyn TextSource> = match source {
            SourceMode::Wikipedia(s) => Box::new(s),
            SourceMode::LocalFile(s) => Box::new(s),
            SourceMode::MistakeList(s) => Box::new(s),
        };
        Ok(Self {
            processor,
            source: source_box,
            mistakes: load_mistakes(),
        })
    }

    async fn run_round(&mut self) -> Result<()> {
        let mut round_results: Vec<(String, String, f64)> = Vec::new();

        // 1. Fetch the next challenge text from the data source (abstracted)
        let target_text = match self.source.get_next_target(&*self.processor).await? {
            Some(t) => t,
            None => {
                println!("No text available to generate a challenge.");
                return Ok(());
            }
        };

        // 2. Normalization and hint display (common logic)
        let judgment_target = self.processor.normalize(&target_text); 

        // Display text is customized for Japanese Mistake List to show spaces for visual separation.
        let display_text = if self.source.language() == "ja" && self.source.display_title().contains("Mistakes List") {
            // Use the raw text with ASCII space separator from get_next_target, 
            // then manually replace ASCII spaces with full-width spaces for clean display.
            target_text.replace('\n', "").replace(' ', "　").trim().to_string()
        } else {
            // For all other modes, the text displayed is the judgment target (what the user should type).
            judgment_target.clone()
        };

        let lang = self.source.language();
        if lang == "ja" {
             // For Japanese, pass the original text for hint display
            self.processor.display_hint(&target_text)?;
        }

        println!("\n--- Input ---\n{}\n↓", display_text); 
        io::stdout().flush()?;

        // 3. Input, time measurement, normalization
        let start = Instant::now();
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim().is_empty() {
            println!("Skipping due to empty line.");
            return Ok(());
        }
        let duration = start.elapsed().as_secs_f64();
        let normalized_input = self.processor.normalize(input.trim()); // normalized_input is spaceless

        // 4. Auto-judgment logic (common)
        // 修正: 3割を超えるトークンがあるばあい評価を実施するロジックを復元
        // 現在は文字数ベースの3分の1 (33%)チェック
        if normalized_input.chars().count() < judgment_target.chars().count() / 3 {
            println!("Input is too short, skipping.");
            return Ok(());
        }

        // NEW: Skip has_matched_token check for Mistake Lists due to concatenation/tokenization issues
        let is_mistake_list = self.source.display_title().contains("Mistakes List");

        if !is_mistake_list { // Only run this check if it's not a mistake list
            if !has_matched_token(&self.processor, &judgment_target, &normalized_input) {
                println!("No input tokens match the expected tokens, skipping.");
                return Ok(());
            }
        }

        // 5. Accumulate results
        round_results.push((judgment_target, normalized_input, duration)); 

        // 6. Finalize stats and process mistake tokens
        let token_mistakes = self.finalize_round_stats(&round_results).await?; // 変更: トークンミス数を取得

        // 7. NEW: Finalize the mistake list if in mistake mode
        self.finalize_mistake_list(token_mistakes);

        Ok(())
    }

    // ADDED: Logic for finalizing MistakeList items
    fn finalize_mistake_list(&mut self, mistake_count: usize) {
        // as_any_mutを使って、ソースがMistakeListSourceかどうかを安全にチェック
        if let Some(mistake_source) = self.source.as_any_mut().downcast_mut::<MistakeListSource>() {
            
            // トークンミスがゼロで、かつ挑戦アイテムが空でない場合のみ、完璧なラウンドと見なす
            if mistake_count == 0 && !mistake_source.current_round_items.is_empty() {
                // 完璧に入力されたので、ラウンドで使用された全アイテムの成功回数をインクリメントする

                // current_round_itemsからアイテムを取り出し、新しいVecにする
                let items_to_update = std::mem::take(&mut mistake_source.current_round_items);

                // 1. 全体の self.mistakes リスト（ファイルと同期）を更新する
                for item_in_round in items_to_update.iter() {
                    // textで照合し、成功回数を増やす
                    if let Some(item_in_all) = self.mistakes.iter_mut().find(|m| m.text == item_in_round.text) {
                        item_in_all.success_count += 1;
                    }
                }

                // 2. MistakeListSource の内部リストを更新（成功回数が5未満のアイテムのみを保持）
                mistake_source.items = self.mistakes.iter()
                    .filter(|i| i.lang == mistake_source.lang && i.success_count < 5)
                    .cloned()
                    .collect();
                
                // 3. ファイルに保存
                mistake_source.save_current_mistakes(); 
            } else {
                // ミスがあった場合、 current_round_items をクリアする
                mistake_source.current_round_items.clear();
            }
        }
    }

    // 変更: 戻り値を usize (token_mistakes) にする
    async fn finalize_round_stats(&mut self, round_results: &Vec<(String, String, f64)>) -> Result<usize> {
        if round_results.is_empty() {
            return Ok(0);
        }

        let mut total_chars = 0;
        let mut mistake_count = 0; // トークンレベルのミス数 (レポート用)
        let mut total_time = 0.0;
        let mut total_mistyped = 0; // 文字レベルのミス数 (Accuracy 計算用)
        let mut total_expected_len = 0;

        for (expected, input, duration) in round_results.iter() {
            self.show_diff(expected, input);

            let expected_tokens = self.processor.tokenize(expected);
            let input_tokens = self.processor.tokenize(input);

            // --- トークンレベルの比較（DPによる最小編集距離アライメント） ---
            let n = expected_tokens.len();
            let m = input_tokens.len();

            // dp[i][j]: Eのi番目までとIのj番目までをアライメントするための最小コスト
            // path[i][j]: 最適なアライメントのための操作 (1:Match/Sub, 2:Del, 3:Ins)
            let mut dp = vec![vec![n + m + 1; m + 1]; n + 1];
            let mut path = vec![vec![0; m + 1]; n + 1];

            // 初期化
            for i in 0..=n {
                dp[i][0] = i; // Deletion (Expected i個を削除)
                if i > 0 { path[i][0] = 2; }
            }
            for j in 0..=m {
                dp[0][j] = j; // Insertion (Input j個を挿入)
                if j > 0 { path[0][j] = 3; }
            }
            dp[0][0] = 0;

            // DPテーブルの構築
            for i in 1..=n {
                for j in 1..=m {
                    let cost = if expected_tokens[i - 1] == input_tokens[j - 1] { 0 } else { 1 };

                    // 1. Substitution/Match
                    let sub_cost = dp[i - 1][j - 1] + cost;
                    // 2. Deletion
                    let del_cost = dp[i - 1][j] + 1;
                    // 3. Insertion
                    let ins_cost = dp[i][j - 1] + 1;

                    if sub_cost <= del_cost && sub_cost <= ins_cost {
                        dp[i][j] = sub_cost;
                        path[i][j] = 1; // Match or Substitution
                    } else if del_cost <= ins_cost {
                        dp[i][j] = del_cost;
                        path[i][j] = 2; // Deletion
                    } else {
                        dp[i][j] = ins_cost;
                        path[i][j] = 3; // Insertion
                    }
                }
            }

            // 最適経路を逆追跡し、ミスと成功を記録
            let mut i = n;
            let mut j = m;
            let mut current_mistakes = 0;
            let mut mistakes_to_add: Vec<String> = Vec::new();
            
            // 後方から見て、Match/Sub/Ins操作に遭遇するまでは、
            // Deletionは「未入力の残り」とみなすフラグ
            let mut untyped_remainder_active = true; 

            while i > 0 || j > 0 {
                match path[i][j] {
                    1 => { // Match or Substitution
                        let et = expected_tokens[i - 1].as_str();
                        if expected_tokens[i - 1] == input_tokens[j - 1] {
                            // Match: 成功として記録
                            // self.source.finalize_token(et); // 削除: Mistake ListモードではGameSessionで一括処理
                        } else {
                            // Substitution: 常にミス
                            mistakes_to_add.push(et.to_string());
                            current_mistakes += 1;
                        }
                        // Match/Sub は入力文字を消費したことを意味するため、Typedエリアに入る
                        untyped_remainder_active = false; 
                        i -= 1; j -= 1;
                    }
                    2 => { // Deletion (Expectedトークンがスキップされた = 誤って入力されなかった)
                        let et = expected_tokens[i - 1].as_str();
                        
                        // FIX: 評価対象エリア内でのみ Deletion をミスとしてカウントする (j > 0 が残っていて、かつ typed areaに入っている場合)
                        if j > 0 && !untyped_remainder_active {
                             mistakes_to_add.push(et.to_string());
                             current_mistakes += 1;
                        }
                        
                        i -= 1;
                    }
                    3 => { // Insertion (Inputに余分なトークンがあった)
                        // Insertion: 常にミス
                        untyped_remainder_active = false; 
                        current_mistakes += 1; 
                        j -= 1;
                    }
                    _ => break,
                }
            }

            // 逆追跡なので、MistakeListに追加するトークンを反転
            mistakes_to_add.reverse();
            mistake_count += current_mistakes;
            for token in mistakes_to_add {
                self.add_mistake_if_new(&token);
            }

            // 2. 文字レベルの比較: Accuracy 統計のための文字レベルエラーを計算する。
            let input_chars_count = input.chars().count();
            
            // --- 文字レベルのアライメント（Accuracy計算用） ---
            
            let e_chars: Vec<char> = expected.chars().collect();
            let i_chars: Vec<char> = input.chars().collect();
            let n_char = e_chars.len(); // n
            let m_char = i_chars.len(); // m

            // dp_char[i][j]: Eのi番目までとIのj番目までをアライメントするための最小コスト
            let mut dp_char = vec![vec![n_char + m_char + 1; m_char + 1]; n_char + 1];
            // path_char[i][j]: 最適なアライメントのための操作 (1:Match/Sub, 2:Del, 3:Ins)
            let mut path_char = vec![vec![0; m_char + 1]; n_char + 1]; 

            // 初期化
            for i in 0..=n_char { 
                dp_char[i][0] = i; 
                if i > 0 { path_char[i][0] = 2; } 
            }
            for j in 0..=m_char { 
                dp_char[0][j] = j; 
                if j > 0 { path_char[0][j] = 3; } 
            }
            dp_char[0][0] = 0;

            // DPテーブルの構築
            for i in 1..=n_char {
                for j in 1..=m_char {
                    let cost = if e_chars[i - 1] == i_chars[j - 1] { 0 } else { 1 };
                    
                    let sub_cost = dp_char[i - 1][j - 1] + cost;
                    let del_cost = dp_char[i - 1][j] + 1;
                    let ins_cost = dp_char[i][j - 1] + 1;

                    if sub_cost <= del_cost && sub_cost <= ins_cost {
                        dp_char[i][j] = sub_cost;
                        path_char[i][j] = 1; // Match/Sub
                    } else if del_cost <= ins_cost {
                        dp_char[i][j] = del_cost;
                        path_char[i][j] = 2; // Deletion
                    } else {
                        dp_char[i][j] = ins_cost;
                        path_char[i][j] = 3; // Insertion
                    }
                }
            }
            
            let char_mistakes_full = dp_char[n_char][m_char]; // Full edit distance

            // FIX: Simple Subtraction for Partial Evaluation
            // 未入力の残りの文字数 (n - m) を最小限のペナルティと見なす
            let untyped_penalty = n_char.saturating_sub(m_char); 
            
            // total_mistyped (Typed Portion Only): フル編集距離から、未入力による最小限のペナルティを減算する
            let char_mistakes = char_mistakes_full.saturating_sub(untyped_penalty);

            // total_expected_len (Accuracyの分母) は、入力文字数 m_char を使用する。
            let expected_chars_for_accuracy = m_char; 

            total_mistyped += char_mistakes; 
            total_chars += input_chars_count;
            total_time += duration;
            total_expected_len += expected_chars_for_accuracy; // <-- m_charを使用
        }

        let wpm = (total_chars as f64 / 5.0) / (total_time / 60.0);
        let kpm = total_chars as f64 / (total_time / 60.0);
        let accuracy = if total_expected_len > 0 {
            // total_mistyped (文字レベルのミス数) を使用して Accuracy を計算
            ((total_expected_len.saturating_sub(total_mistyped)) as f64 / total_expected_len as f64) * 100.0
        } else {
            0.0
        };

        if total_expected_len > 0 {
            println!("== Individual Report ==");
            // mistake_count (トークンレベルのミス数) を使用して Mistakes を報告
            if accuracy < 100.0 {
                println!("Accuracy: {:.1}%, Mistakes: {}", accuracy, mistake_count);
            } else {
                println!("Accuracy: {:.1}%, Perfect!", accuracy);
            }
        }

        println!("== Overall Result ==");
        println!("Speed WPM: {:.1}, KPM: {:.1}", wpm, kpm);

        Ok(mistake_count)
    }

    fn add_mistake_if_new(&mut self, token: &str) {
        if !self.mistakes.iter().any(|m| m.text == token) {
            let lang = self.source.language().to_string();
            let lang = if lang == "ja" { "ja".to_string() } else { "en".to_string() };
            self.mistakes.push(MistakeItem {
                text: token.to_string(),
                lang,
                success_count: 0,
            });
            save_mistakes(&self.mistakes);
        }
    }

    // --- show_diff (トークンDPアライメントによる修正) ---
    // トークン単位でDPを実行し、結果をトークン単位で表示します。
    fn show_diff(&self, expected: &str, input: &str) {
        println!("\n== Difference Display (Token-Unit) ==");
        
        let expected_tokens = self.processor.tokenize(expected);
        let input_tokens = self.processor.tokenize(input);
        
        let n = expected_tokens.len();
        let m = input_tokens.len();
        // 日本語は全角スペース、英語は半角スペース
        let separator = if self.source.language() == "ja" { "　" } else { " " };
        
        // DP table for cost (Edit Distance)
        let mut dp = vec![vec![n + m + 1; m + 1]; n + 1];
        // Path table (1: Match/Sub, 2: Del, 3: Ins)
        let mut path = vec![vec![0; m + 1]; n + 1];

        // Initialization
        for i in 0..=n { dp[i][0] = i; if i > 0 { path[i][0] = 2; } } // Deletion
        for j in 0..=m { dp[0][j] = j; if j > 0 { path[0][j] = 3; } } // Insertion
        dp[0][0] = 0;

        // Fill DP table
        for i in 1..=n {
            for j in 1..=m {
                let cost = if expected_tokens[i - 1] == input_tokens[j - 1] { 0 } else { 1 };
                
                // 1. Substitution/Match
                let sub_cost = dp[i - 1][j - 1] + cost;
                // 2. Deletion
                let del_cost = dp[i - 1][j] + 1;
                // 3. Insertion
                let ins_cost = dp[i][j - 1] + 1;

                if sub_cost <= del_cost && sub_cost <= ins_cost {
                    dp[i][j] = sub_cost;
                    path[i][j] = 1; 
                } else if del_cost <= ins_cost {
                    dp[i][j] = del_cost;
                    path[i][j] = 2; 
                } else {
                    dp[i][j] = ins_cost;
                    path[i][j] = 3; 
                }
            }
        }

        // Backtrack and build the colored strings
        let mut i = n;
        let mut j = m;
        let mut out_e_vec: Vec<String> = Vec::new();
        let mut out_i_vec: Vec<String> = Vec::new();
        
        // Missing token placeholder
        let placeholder = if self.source.language() == "ja" { "＿" } else { "___" }; 
        
        // Diff表示でも未入力部分をグレーアウトするために、stateful flagを使用
        let mut untyped_remainder_active = true;
        
        // NEW: Count the number of gray tokens displayed
        let mut gray_token_count = 0;
        const MAX_GRAY_TOKENS: usize = 10; // Max number of gray tokens to display

        while i > 0 || j > 0 {
            let op = path[i][j];

            match op {
                1 => { // Match or Substitution
                    // 入力があったため、typed areaに入る
                    untyped_remainder_active = false;
                    let et = expected_tokens[i - 1].as_str();
                    let it = input_tokens[j - 1].as_str();
                    
                    if et == it {
                        // Match (一致)
                        out_e_vec.push(color!(et, "green"));
                        out_i_vec.push(color!(it, "green"));
                    } else {
                        // Substitution (置換)
                        out_e_vec.push(color!(et, "bold_red"));
                        out_i_vec.push(color!(it, "bold_red"));
                    }
                    i -= 1; j -= 1;
                }
                2 => { // Deletion (ExpectedはあるがInputにない)
                    let et = expected_tokens[i - 1].as_str();
                    
                    if j > 0 && !untyped_remainder_active { 
                        // Typed Deletion (ミス) - 評価エリア内での欠落は赤
                        out_e_vec.push(color!(et, "bold_red"));
                        out_i_vec.push(color!(placeholder, "bold_red")); 
                        i -= 1;
                    } else {
                        // Untyped Remainder (グレーアウト) - j=0または未入力エリア
                        if gray_token_count < MAX_GRAY_TOKENS {
                            out_e_vec.push(color!(et, "gray"));
                            out_i_vec.push(color!(placeholder, "gray")); 
                            gray_token_count += 1;
                        } else if gray_token_count == MAX_GRAY_TOKENS {
                            // 11個目以降は省略記号(...)に置き換える
                            let ellipsis = if self.source.language() == "ja" { "..." } else { "..." };
                            out_e_vec.push(color!(ellipsis, "gray"));
                            out_i_vec.push(color!(ellipsis, "gray"));
                            gray_token_count += 1;
                        }
                        // If gray_token_count > MAX_GRAY_TOKENS, we just skip adding to the vecs.
                        i -= 1;
                    }
                }
                3 => { // Insertion (InputにはあるがExpectedにない)
                    // 入力があったため、typed areaに入る
                    untyped_remainder_active = false;
                    let it = input_tokens[j - 1].as_str();
                    
                    // Input側は赤で表示し、Expected側はプレースホルダー
                    out_e_vec.push(color!(placeholder, "bold_red"));
                    out_i_vec.push(color!(it, "bold_red"));
                    j -= 1;
                }
                _ => break,
            }
        }
        
        // 逆順なので反転して出力
        out_e_vec.reverse();
        out_i_vec.reverse();

        // トークン単位表示では、ExpectedとInputが同じ行でアライメントされる
        println!("Expected:\n{}", out_e_vec.join(separator));
        println!("Input:\n{}", out_i_vec.join(separator));
    }
}

// --- Main Function and Source Selection Logic ---

fn select_source(last: Option<SourceMode>) -> Option<SourceMode> {
    println!("1 - ja, Wikipedia (Japanese)");
    println!("2 - en, Wikipedia (English)");
    println!("3 - Japanese Mistakes List");
    println!("4 - English Mistakes List");

    let mut file_options: Vec<(String, String, String, usize, Vec<String>)> = Vec::new();

    if let Ok(dir) = fs::read_dir(".") {
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().map(|ext| ext == "txt").unwrap_or(false) {
                if let Ok(content) = fs::read_to_string(&path) {
                    let mut lines = content.lines();
                    if let Some(header) = lines.next() {
                        let parts: Vec<&str> = header.split(',').collect();
                        if parts.len() >= 3 {
                            let lang = parts[0].trim().to_string();
                            let token_count = std::cmp::max(parts[1].trim().parse::<usize>().unwrap_or(5), 5);
                            let file_title = parts[2].trim().to_string();
                            let all_lines: Vec<String> = lines
                                .map(|l| l.trim().to_string())
                                .filter(|l| !l.is_empty())
                                .collect();
                            if !all_lines.is_empty() {
                                let display_title = format!("{} - File: {}", file_title, path.file_stem().unwrap().to_string_lossy());
                                file_options.push((lang, file_title, display_title, token_count, all_lines));
                            }
                        }
                    }
                }
            }
        }
    }

    for (idx, (_, _, display_title, _, _)) in file_options.iter().enumerate() {
        println!("{} - {}", idx + 5, display_title);
    }

    println!("Enter 'q' to quit");

    print!("Enter number > ");
    io::stdout().flush().ok()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input).ok()?;

    if input.trim() == "q" {
        return None;
    }

    match input.trim().parse::<usize>() {
        Ok(idx) => match idx {
            1 => Some(SourceMode::Wikipedia(WikipediaSource::new("ja".to_string()))),
            2 => Some(SourceMode::Wikipedia(WikipediaSource::new("en".to_string()))),
            3 => Some(SourceMode::MistakeList(MistakeListSource::new("ja".to_string()))),
            4 => Some(SourceMode::MistakeList(MistakeListSource::new("en".to_string()))),
            n if n >= 5 && n < 5 + file_options.len() => {
                let (lang, file_title, _, token_count, all_lines) = file_options.remove(n - 5);
                Some(SourceMode::LocalFile(LocalFileSource {
                    lang,
                    title: file_title,
                    token_count,
                    all_lines,
                }))
            }
            // 無効な番号が入力された場合、前回のソースをリフレッシュして再利用
            _ => last.map(|s| s.refresh()),
        },
        // Enterキーのみが入力された場合、前回のソースをリフレッシュして再利用
        _ => last.map(|s| s.refresh()),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let package_name = env!("CARGO_PKG_NAME");
    let version = env!("CARGO_PKG_VERSION");
    let authors = env!("CARGO_PKG_AUTHORS");

    println!("{} v{}", package_name, version);
    println!("Copyright (C)2025 by {}", authors);
    println!("Wikipedia texts are licenced under CC BY-SA 3.0\n");

    let mut last_source: Option<SourceMode> = None;
    loop {
        // If last_source is Some, pass a clone to select_source
        let source = match select_source(last_source.as_ref().map(|s| s.clone())) {
            Some(s) => s,
            None => break,
        };

        // Use SourceMode::display_title() before creating GameSession
        println!("\n=== Starting {} Mode ===", source.display_title());
        
        let mut session = GameSession::new(source)?;


        if let Err(e) = session.run_round().await {
            eprintln!("Error: {}", e);
        }
        
        // If the session was successful, update last_source with the current source's state
        // clone_source()でBox<dyn TextSource>からSourceModeに戻す
        last_source = Some(session.source.clone_source());
        println!("\n------------------------");
    }
    Ok(())
}