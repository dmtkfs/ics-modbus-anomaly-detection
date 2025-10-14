import pandas as pd
import numpy as np
import pickle
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

# Create necessary directories
os.makedirs('data/baselines', exist_ok=True)
os.makedirs('figures/heuristics', exist_ok=True)

class ModbusHeuristicsDetector:
    def __init__(self):
        self.baseline_write_ratios = {}
        self.baseline_fc_distributions = {}
        self.write_codes = [5, 6, 15, 16]  # Write function codes
        
    def load_data(self, filepath='master.csv'):
        """Step 1: Load merged dataset"""
        print("Step 1: Loading dataset...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        return df
    
    def compute_baselines(self, df):
        """Step 2: Compute baseline statistics from benign traffic"""
        print("Step 2: Computing baselines...")
        benign_df = df[df['Label'] == 'Benign'].copy()
        
        # H1 Baseline: Write ratios per source
        for source in benign_df['Source'].unique():
            source_data = benign_df[benign_df['Source'] == source]
            write_count = len(source_data[source_data['FunctionCodeNum'].isin(self.write_codes)])
            total_count = len(source_data)
            write_ratio = write_count / total_count if total_count > 0 else 0
            self.baseline_write_ratios[source] = write_ratio
        
        # H2 Baseline: Function code distributions per source
        for source in benign_df['Source'].unique():
            source_data = benign_df[benign_df['Source'] == source]
            fc_counts = source_data['FunctionCodeNum'].value_counts()
            fc_freq = fc_counts / fc_counts.sum()
            self.baseline_fc_distributions[source] = fc_freq
        
        # Save baselines
        with open('data/baselines/baseline_write_ratios.pkl', 'wb') as f:
            pickle.dump(self.baseline_write_ratios, f)
        with open('data/baselines/baseline_fc_distributions.pkl', 'wb') as f:
            pickle.dump(self.baseline_fc_distributions, f)
        
        print(f"Computed baselines for {len(self.baseline_write_ratios)} sources")
    
    def detect_h1_write_spikes(self, df, threshold_multiplier=3.0):
        """Step 3: H1 - Write-Rate Spike Detection"""
        print("Step 3: Applying H1 (Write-Rate Spike)...")
        h1_flags = []
        
        # Calculate global baseline stats for sources without individual baselines
        all_write_ratios = list(self.baseline_write_ratios.values())
        global_mean = np.mean(all_write_ratios)
        global_std = np.std(all_write_ratios)
        global_threshold = global_mean + threshold_multiplier * global_std
        
        for _, row in df.iterrows():
            source = row['Source']
            
            # Get current window data for this source
            source_data = df[df['Source'] == source]
            write_count = len(source_data[source_data['FunctionCodeNum'].isin(self.write_codes)])
            total_count = len(source_data)
            current_ratio = write_count / total_count if total_count > 0 else 0
            
            # Check against baseline
            if source in self.baseline_write_ratios:
                baseline_ratio = self.baseline_write_ratios[source]
                # Use 3-sigma rule: μ + 3σ
                threshold = baseline_ratio + threshold_multiplier * baseline_ratio
                is_anomaly = current_ratio > threshold
            else:
                # Use global threshold for unknown sources
                is_anomaly = current_ratio > global_threshold
            
            h1_flags.append(1 if is_anomaly else 0)
        
        return h1_flags
    
    def detect_h2_variant_f(self, df, k_threshold=2.0):
        """Step 4: H2 Variant F - Composite Hybrid (Frequency + Directionality)"""
        print("Step 4: Applying H2 Variant F (Composite Hybrid)...")
        h2_flags = []
        
        for _, row in df.iterrows():
            source = row['Source']
            fc_num = row['FunctionCodeNum']
            dest_port = row.get('Destination Port', 0)
            src_port = row.get('Source Port', 0)
            
            is_anomaly = False
            
            # Part 1: Frequency Outlier Detection (Variant E)
            if source in self.baseline_fc_distributions:
                baseline_freq = self.baseline_fc_distributions[source]
                
                # Get current source's function code distribution
                source_data = df[df['Source'] == source]
                current_fc_counts = source_data['FunctionCodeNum'].value_counts()
                current_fc_freq = current_fc_counts / current_fc_counts.sum()
                
                # Check if this function code frequency is anomalous
                if fc_num in baseline_freq.index and fc_num in current_fc_freq.index:
                    baseline_fc_freq = baseline_freq[fc_num]
                    current_fc_freq_val = current_fc_freq[fc_num]
                    
                    # Simple z-score approximation
                    if baseline_fc_freq > 0:
                        deviation = abs(current_fc_freq_val - baseline_fc_freq) / baseline_fc_freq
                        if deviation > k_threshold:
                            is_anomaly = True
                elif fc_num not in baseline_freq.index:
                    # Function code never seen in baseline = anomaly
                    is_anomaly = True
            
            # Part 2: Directionality Check (Variant D)
            # Check for master/slave role violations
            if dest_port == 502:  # This source is acting as master
                # Count how often this source acts as master vs slave
                source_data = df[df['Source'] == source]
                master_actions = len(source_data[source_data['Destination Port'] == 502])
                slave_actions = len(source_data[source_data['Source Port'] == 502])
                
                # If predominantly acts as master but occasionally as slave, flag it
                if master_actions > 0 and slave_actions > 0:
                    master_ratio = master_actions / (master_actions + slave_actions)
                    if 0.1 < master_ratio < 0.9:  # Mixed behavior is suspicious
                        is_anomaly = True
            
            h2_flags.append(1 if is_anomaly else 0)
        
        return h2_flags
    
    def evaluate_heuristics(self, df, h1_flags, h2_flags):
        """Step 5: Evaluate individual and combined heuristic performance"""
        print("Step 5: Evaluating heuristics...")
        
        # Convert labels to binary
        true_labels = (df['Label'] != 'Benign').astype(int)
        
        # Combined heuristics (logical OR)
        combined_flags = [1 if (h1 or h2) else 0 for h1, h2 in zip(h1_flags, h2_flags)]
        
        # Calculate metrics
        results = {}
        
        # H1 metrics
        h1_precision = precision_score(true_labels, h1_flags, zero_division=0)
        h1_recall = recall_score(true_labels, h1_flags, zero_division=0)
        h1_f1 = f1_score(true_labels, h1_flags, zero_division=0)
        results['H1'] = {'precision': h1_precision, 'recall': h1_recall, 'f1': h1_f1}
        
        # H2 metrics
        h2_precision = precision_score(true_labels, h2_flags, zero_division=0)
        h2_recall = recall_score(true_labels, h2_flags, zero_division=0)
        h2_f1 = f1_score(true_labels, h2_flags, zero_division=0)
        results['H2'] = {'precision': h2_precision, 'recall': h2_recall, 'f1': h2_f1}
        
        # Combined metrics
        combined_precision = precision_score(true_labels, combined_flags, zero_division=0)
        combined_recall = recall_score(true_labels, combined_flags, zero_division=0)
        combined_f1 = f1_score(true_labels, combined_flags, zero_division=0)
        results['Combined'] = {'precision': combined_precision, 'recall': combined_recall, 'f1': combined_f1}
        
        # Print results
        for method, metrics in results.items():
            print(f"{method}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        return results
    
    def visualize_results(self, df, h1_flags, h2_flags, results):
        """Step 6: Create visualizations"""
        print("Step 6: Creating visualizations...")
        
        # Performance comparison bar chart
        methods = list(results.keys())
        precisions = [results[method]['precision'] for method in methods]
        recalls = [results[method]['recall'] for method in methods]
        f1s = [results[method]['f1'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Heuristic Method')
        ax.set_ylabel('Score')
        ax.set_title('Heuristics Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/heuristics/performance_comparison.png', dpi=300)
        plt.close()
        
        # Per-attack family analysis
        if 'AttackFamily' in df.columns:
            attack_families = df[df['Label'] != 'Benign']['AttackFamily'].unique()
            family_recalls = {}
            
            for family in attack_families:
                family_mask = df['AttackFamily'] == family
                family_true = (df[family_mask]['Label'] != 'Benign').astype(int)
                family_h1 = [h1_flags[i] for i in range(len(df)) if family_mask.iloc[i]]
                family_h2 = [h2_flags[i] for i in range(len(df)) if family_mask.iloc[i]]
                family_combined = [1 if (h1 or h2) else 0 for h1, h2 in zip(family_h1, family_h2)]
                
                if len(family_true) > 0:
                    h1_recall = recall_score(family_true, family_h1, zero_division=0)
                    h2_recall = recall_score(family_true, family_h2, zero_division=0)
                    combined_recall = recall_score(family_true, family_combined, zero_division=0)
                    
                    family_recalls[family] = {
                        'H1': h1_recall,
                        'H2': h2_recall,
                        'Combined': combined_recall
                    }
            
            # Plot per-family recall
            families = list(family_recalls.keys())
            h1_recalls = [family_recalls[f]['H1'] for f in families]
            h2_recalls = [family_recalls[f]['H2'] for f in families]
            combined_recalls = [family_recalls[f]['Combined'] for f in families]
            
            x = np.arange(len(families))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width, h1_recalls, width, label='H1', alpha=0.8)
            ax.bar(x, h2_recalls, width, label='H2', alpha=0.8)
            ax.bar(x + width, combined_recalls, width, label='Combined', alpha=0.8)
            
            ax.set_xlabel('Attack Family')
            ax.set_ylabel('Recall')
            ax.set_title('Recall by Attack Family')
            ax.set_xticks(x)
            ax.set_xticklabels(families, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('figures/heuristics/recall_by_attack_family.png', dpi=300)
            plt.close()
    
    def export_results(self, df, h1_flags, h2_flags):
        """Step 7: Export results for ML team integration"""
        print("Step 7: Exporting results...")
        
        # Create results dataframe
        results_df = df[['Source', 'Label', 'AttackFamily']].copy()
        results_df['H1_flag'] = h1_flags
        results_df['H2_flag'] = h2_flags
        results_df['Combined_flag'] = [1 if (h1 or h2) else 0 for h1, h2 in zip(h1_flags, h2_flags)]
        
        # Save to CSV
        results_df.to_csv('data/processed/heuristics_results.csv', index=False)
        
        # Save metrics to CSV (EIP compliance)
        metrics_data = {
            'model': ['H1_WriteSpike', 'H2_VariantF', 'H1_H2_Combined'],
            'precision': [
                precision_score((df['Label'] != 'Benign').astype(int), h1_flags, zero_division=0),
                precision_score((df['Label'] != 'Benign').astype(int), h2_flags, zero_division=0),
                precision_score((df['Label'] != 'Benign').astype(int), 
                               [1 if (h1 or h2) else 0 for h1, h2 in zip(h1_flags, h2_flags)], 
                               zero_division=0)
            ],
            'recall': [
                recall_score((df['Label'] != 'Benign').astype(int), h1_flags, zero_division=0),
                recall_score((df['Label'] != 'Benign').astype(int), h2_flags, zero_division=0),
                recall_score((df['Label'] != 'Benign').astype(int), 
                            [1 if (h1 or h2) else 0 for h1, h2 in zip(h1_flags, h2_flags)], 
                            zero_division=0)
            ],
            'f1': [
                f1_score((df['Label'] != 'Benign').astype(int), h1_flags, zero_division=0),
                f1_score((df['Label'] != 'Benign').astype(int), h2_flags, zero_division=0),
                f1_score((df['Label'] != 'Benign').astype(int), 
                        [1 if (h1 or h2) else 0 for h1, h2 in zip(h1_flags, h2_flags)], 
                        zero_division=0)
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        # Round to 3 decimals (EIP requirement)
        for col in ['precision', 'recall', 'f1']:
            metrics_df[col] = metrics_df[col].round(3)
        
        metrics_df.to_csv('figures/heuristics/metrics.csv', index=False)
        
        print("Results exported successfully!")
        return results_df

# Main execution function
def run_complete_heuristics():
    """Execute the complete 7-step heuristics workflow"""
    detector = ModbusHeuristicsDetector()
    
    # Execute all steps
    df = detector.load_data('master.csv')  # Replace with your file path
    detector.compute_baselines(df)
    h1_flags = detector.detect_h1_write_spikes(df)
    h2_flags = detector.detect_h2_variant_f(df)
    results = detector.evaluate_heuristics(df, h1_flags, h2_flags)
    detector.visualize_results(df, h1_flags, h2_flags, results)
    final_results = detector.export_results(df, h1_flags, h2_flags)
    
    print("\n=== COMPLETE HEURISTICS WORKFLOW FINISHED ===")
    print("✓ H1 (Write-Rate Spike) implemented")
    print("✓ H2 (Function-Code Anomaly - Variant F) implemented") 
    print("✓ Combined heuristics evaluated")
    print("✓ Results exported for ML team integration")
    print("✓ All files ready for EIP compliance")
    
    return final_results

# Run the complete system
if __name__ == "__main__":
    results = run_complete_heuristics()
