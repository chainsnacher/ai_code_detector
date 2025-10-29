"""Quick test to verify the system works."""
import sys
sys.path.append('src')

try:
    from preprocessing.language_detector import LanguageDetector
    print("✓ Language detector imported")
    
    detector = LanguageDetector()
    lang, conf = detector.detect_language("def hello(): pass")
    print(f"✓ Language detection working: {lang} ({conf:.2f})")
except Exception as e:
    print(f"✗ Language detector failed: {e}")

try:
    from utils.powerbi_exporter import PowerBIExporter
    print("✓ Power BI exporter imported")
except Exception as e:
    print(f"✗ Power BI exporter failed: {e}")

try:
    # Try to import with data_labeler manually
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_labeler", "src/utils/data_labeler.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("✓ Data labeler imported via importlib")
except Exception as e:
    print(f"✗ Data labeler failed: {e}")
    import traceback
    traceback.print_exc()

print("\nSimple imports working. Running main project now...")

