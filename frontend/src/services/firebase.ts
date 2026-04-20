import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyCn9MCTLcm9j1goTxDPdlpSoszKvVVd4_0",
  authDomain: "face-recognition-attenda-311d6.firebaseapp.com",
  projectId: "face-recognition-attenda-311d6",
  storageBucket: "face-recognition-attenda-311d6.firebasestorage.app",
  messagingSenderId: "866751705112",
  appId: "1:866751705112:web:ba2f5e6c72f59cceedd36d",
};

export const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();