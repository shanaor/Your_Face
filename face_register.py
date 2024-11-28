import face_recognition
import cv2
import os
import pickle
import json
import time
import numpy as np
import urllib.request

class FaceAuthSystem:
    def __init__(self):
        self.data_dir = "face_data"
        self.user_data_file = os.path.join(self.data_dir, "user_data.json")
        self.banned_file = os.path.join(self.data_dir, "banned_users.json")
        self.face_confidence_threshold = 0.6
        
        banned_image_path = 'angry_face.png'  # Change to your image path or URL
        if banned_image_path.startswith(('http://', 'https://')):
            resp = urllib.request.urlopen(banned_image_path)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            self.banned_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        else:
            self.banned_image = cv2.imread(banned_image_path, cv2.IMREAD_UNCHANGED)
        
        os.makedirs(self.data_dir, exist_ok=True)
        self._initialize_data()

    def _initialize_data(self):
        for file in [self.user_data_file, self.banned_file]:
            if not os.path.exists(file):
                with open(file, 'w') as f:
                    json.dump({}, f)

    def show_countdown(self, frame, seconds):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Starting in: {seconds}", (w//2 - 100, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow('Face Detection', frame)
        cv2.waitKey(1)

    def show_banned_screen(self):
        if self.banned_image is not None:
            # Resize banned image if needed
            height, width = 400, 600
            banned_img_resized = cv2.resize(self.banned_image, (width, height))
            cv2.imshow('BANNED', banned_img_resized)
        
        banned_screen = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(banned_screen, "YOUR FACE IS BANNED!!!!", (50, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow('BANNED', banned_screen)
        cv2.waitKey(2000)
        cv2.destroyWindow('BANNED')

    def process_frame(self, frame, process_type="register", username=None):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        for (top, right, bottom, left) in face_locations:
            top *= 4; right *= 4; bottom *= 4; left *= 4
            color = (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        if process_type == "register":
            cv2.putText(frame, "Registration Mode - Press 'c' to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Login Mode - Verifying...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if username:
            cv2.putText(frame, f"User: {username}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame, face_locations

    def check_if_banned(self, face_encoding):
        with open(self.banned_file, 'r') as f:
            banned_users = json.load(f)
        
        for user_data in banned_users.values():
            try:
                with open(user_data['face_file'], 'rb') as f:
                    saved_encoding = pickle.load(f)
                    distance = face_recognition.face_distance([saved_encoding], face_encoding)[0]
                    if distance < self.face_confidence_threshold:
                        return True
            except Exception:
                continue
        return False

    def register_user(self):
        name = input("Enter username to register: ").strip()
        if not name or not name.isalnum():
            print("Invalid username. Use only letters and numbers.")
            return False

        with open(self.user_data_file, 'r') as f:
            user_data = json.load(f)
            if name in user_data:
                print("Username already exists.")
                return False

        cap = cv2.VideoCapture(0)
        face_encoding = None
        
        for i in range(5, 0, -1):
            ret, frame = cap.read()
            if ret:
                self.show_countdown(frame, i)
            time.sleep(1)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, face_locations = self.process_frame(frame, "register", name)
            cv2.imshow('Face Registration', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('c') and len(face_locations) == 1:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encoding = face_recognition.face_encodings(rgb_frame)[0]
                break

        cap.release()
        cv2.destroyAllWindows()

        if face_encoding is not None:
            face_file = os.path.join(self.data_dir, f"{name}_face.pkl")
            with open(face_file, "wb") as f:
                pickle.dump(face_encoding, f)

            user_data[name] = {
                "face_file": face_file
            }
            
            with open(self.user_data_file, 'w') as f:
                json.dump(user_data, f)

            print(f"Successfully registered {name}")
            return True
        
        print("Registration failed")
        return False

    def ban_user(self):
        with open(self.user_data_file, 'r') as f:
            users = json.load(f)
        
        if not users:
            print("No registered users found")
            return

        print("\nRegistered users:")
        for i, username in enumerate(users.keys(), 1):
            print(f"{i}. {username}")

        try:
            choice = int(input("\nEnter number of user to ban: ")) - 1
            username = list(users.keys())[choice]
            
            with open(self.banned_file, 'r') as f:
                banned_users = json.load(f)
            
            banned_users[username] = users[username]
            with open(self.banned_file, 'w') as f:
                json.dump(banned_users, f)
            
            del users[username]
            with open(self.user_data_file, 'w') as f:
                json.dump(users, f)
            
            print(f"User {username} has been banned")
        except (ValueError, IndexError):
            print("Invalid selection")

    def login_user(self):
        cap = cv2.VideoCapture(0)
        login_successful = False
        matched_user = None
        
        for i in range(5, 0, -1):
            ret, frame = cap.read()
            if ret:
                self.show_countdown(frame, i)
            time.sleep(1)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame)
                if face_encodings:
                    is_banned = self.check_if_banned(face_encodings[0])
                    if is_banned:
                        self.show_banned_screen()
                        cap.release()
                        cv2.destroyAllWindows()
                        return None
                    else:
                        with open(self.user_data_file, 'r') as f:
                            user_data = json.load(f)
                        for username, data in user_data.items():
                            with open(data['face_file'], 'rb') as f:
                                saved_encoding = pickle.load(f)
                                distance = face_recognition.face_distance([saved_encoding], face_encodings[0])[0]
                                if distance < self.face_confidence_threshold:
                                    login_successful = True
                                    matched_user = username
                                    break

            frame, _ = self.process_frame(frame, "login", matched_user)
            cv2.imshow('Face Login', frame)

            if login_successful:
                cv2.waitKey(1000)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return matched_user

def main():
    system = FaceAuthSystem()
    
    while True:
        print("\n1. Register New User")
        print("2. Login")
        print("3. Exit")
        print("4. Ban User")
        
        try:
            choice = input("Select option (1-4): ").strip()
            
            if choice == "1":
                system.register_user()
            elif choice == "2":
                user = system.login_user()
                if user:
                    print(f"Login successful! Welcome {user}")
                else:
                    print("Login failed")
            elif choice == "3":
                print("Goodbye!")
                break
            elif choice == "4":
                system.ban_user()
            else:
                print("Invalid choice")
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()