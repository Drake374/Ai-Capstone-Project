import { useNavigate } from "react-router-dom";
import { loginWithGoogle } from "../services/authService";
import "./LoginPage.css";

const LoginPage = () => {
  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const user = await loginWithGoogle();

      const userData = {
        name: user.displayName,
        email: user.email,
        photo: user.photoURL,
      };

      localStorage.setItem("user", JSON.stringify(userData));
      navigate("/");
    } catch (error) {
      console.error(error);
      alert("Login failed. Please try again.");
    }
  };

  return (
    <div className="login-page">
      <div className="login-overlay">
        <header className="login-header">
          <div className="brand-box">
            <div className="brand-title">CENTENNIAL</div>
            <div className="brand-subtitle">COLLEGE</div>
          </div>
        </header>

        <main className="login-main">
          <div className="login-card">
            <div className="accent-bar"></div>

            <h1>AI Attendance Portal</h1>
            <p className="login-description">
              Sign in using your Google account to simulate Centennial access.
            </p>

            <button className="google-login-btn" onClick={handleLogin}>
              Sign in with Google
            </button>

            <p className="login-note">
              Students can access face registration and attendance.
              <br />
              Admin and instructors can review and download logs.
            </p>
          </div>
        </main>
      </div>
    </div>
  );
};

export default LoginPage;