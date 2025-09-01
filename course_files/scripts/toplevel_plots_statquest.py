#!/usr/bin/env python3
"""
Master script to generate all t-SNE explanation diagrams for biologists.

This script will create all 7 figures explaining t-SNE step by step.
Each figure is saved as both PNG (for presentations) and PDF (for publications).

Requirements:
pip install matplotlib numpy pandas scikit-learn

Optional for animation:
pip install pillow

Author: Created for biologists to understand t-SNE visually
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def setup_matplotlib():
    """Configure matplotlib for high-quality output"""
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

def run_all_diagrams():
    """Run all t-SNE explanation diagrams"""
    
    setup_matplotlib()
    
    print("🧬 Generating t-SNE explanation diagrams for biologists...")
    print("=" * 60)
    
    # Import individual diagram functions
    from tsne_step1_highdim_problem import create_highdim_problem_figure
    from tsne_step2_similarity import create_similarity_figure  
    from tsne_step3_random_placement import create_random_placement_figure
    from tsne_step4_optimization import create_optimization_figure, create_animation
    from tsne_step5_final_result import create_final_result_figure
    from tsne_step6_pca_comparison import create_pca_comparison_figure
    from tsne_step7_pitfalls import create_pitfalls_figure
    
    figures = [
        ("Step 1: High-Dimensional Data Problem", create_highdim_problem_figure),
        ("Step 2: Similarity Calculation", create_similarity_figure),
        ("Step 3: Random Initial Placement", create_random_placement_figure), 
        ("Step 4: Iterative Optimization", create_optimization_figure),
        ("Step 5: Final Result & Interpretation", create_final_result_figure),
        ("Step 6: Comparison with PCA", create_pca_comparison_figure),
        ("Step 7: Pitfalls & Best Practices", create_pitfalls_figure)
    ]
    
    for i, (title, func) in enumerate(figures, 1):
        print(f"📊 Creating Figure {i}: {title}")
        try:
            func()
            print(f"✅ Successfully created Figure {i}")
        except Exception as e:
            print(f"❌ Error creating Figure {i}: {str(e)}")
        print()
    
    # Optional: Create animation
    try:
        print("🎬 Creating optimization animation...")
        create_animation()
        print("✅ Animation created successfully")
    except ImportError:
        print("⚠️ Skipping animation (requires pillow: pip install pillow)")
    except Exception as e:
        print(f"❌ Error creating animation: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎉 All diagrams generated successfully!")
    print("\nFiles created:")
    print("• tsne_step1_highdim_problem.png/pdf")
    print("• tsne_step2_similarity_calculation.png/pdf") 
    print("• tsne_step3_random_placement.png/pdf")
    print("• tsne_step4_optimization_stages.png/pdf")
    print("• tsne_step5_final_result_interpretation.png/pdf")
    print("• tsne_step6_comparison_with_pca.png/pdf")
    print("• tsne_step7_pitfalls_best_practices.png/pdf")
    print("• tsne_optimization_animation.gif (if pillow installed)")
    
    print("\n📚 How to use these figures:")
    print("• Use PNG files for PowerPoint presentations")
    print("• Use PDF files for LaTeX documents and publications") 
    print("• Show figures sequentially to build understanding")
    print("• Adapt the biological examples to your specific data")

if __name__ == "__main__":
    # For standalone execution, create a simple version of each diagram
    print("🚀 Creating t-SNE explanation diagrams...")
    
    # You can copy and paste each individual diagram code here,
    # or import them as separate modules
    
    # For demonstration, here's a simplified version that creates all diagrams:
    
    exec(open('tsne_step1_highdim_problem.py').read())
    print("✅ Step 1 complete")
    
    exec(open('tsne_step2_similarity_calculation.py').read()) 
    print("✅ Step 2 complete")
    
    exec(open('tsne_step3_random_placement.py').read())
    print("✅ Step 3 complete")
    
    exec(open('tsne_step4_optimization.py').read())
    print("✅ Step 4 complete")
    
    exec(open('tsne_step5_final_result_interpretation.py').read())
    print("✅ Step 5 complete")
    
    exec(open('tsne_step6_comparison_with_pca.py').read())
    print("✅ Step 6 complete")
    
    exec(open('tsne_step7_pitfalls_best_practices.py').read())
    print("✅ Step 7 complete")
    
    print("\n🎉 All diagrams created successfully!")

# Example usage in Jupyter notebook:
"""
# Run individual diagrams:
exec(open('tsne_step1_highdim_problem.py').read())

# Or run all at once:
exec(open('run_all_tsne_diagrams.py').read())

# Or import as modules:
import tsne_step1_highdim_problem
import tsne_step2_similarity_calculation
# etc.
"""