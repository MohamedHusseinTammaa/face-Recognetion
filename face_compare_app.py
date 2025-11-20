from deepface import DeepFace
import cv2
import os

def compare_faces(img1_path, img2_path):
    try:
        # Check if files exist
        if not os.path.exists(img1_path):
            print(f"❌ Error: Image 1 not found at {img1_path}")
            return
        if not os.path.exists(img2_path):
            print(f"❌ Error: Image 2 not found at {img2_path}")
            return
        
        print("📸 Loading images...")
        print(f"Image 1: {img1_path}")
        print(f"Image 2: {img2_path}")
        
        # Verify images can be read
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None:
            print(f"❌ Error: Cannot read image 1")
            return
        if img2 is None:
            print(f"❌ Error: Cannot read image 2")
            return
        
        print("🔍 Comparing faces...")
        
        # Perform face verification with error handling
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True
        )
        
        print("\n" + "="*50)
        if result["verified"]:
            print("✅ SAME PERSON detected!")
        else:
            print("❌ NOT the same person.")
        print("="*50)
        print(f"Distance: {result['distance']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")
        print(f"Model: {result['model']}")
        print(f"Detector: {result['detector_backend']}")
        print("="*50)
        
        return result
        
    except ValueError as e:
        print(f"❌ Face Detection Error: {e}")
        print("\n💡 Tips:")
        print("   - Make sure faces are clearly visible in both images")
        print("   - Try setting enforce_detection=False in the code")
        print("   - Try a different detector_backend: 'mtcnn', 'retinaface', 'ssd'")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    # Update these paths to your actual image locations
    img1 = r"E:\ai\face\Screenshot 2025-11-20 150548.png"
    img2 = r"E:\ai\face\Screenshot 2025-03-23 212251.png"
    
    compare_faces(img1, img2)