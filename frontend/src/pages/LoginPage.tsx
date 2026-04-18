import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { loginWithGoogle } from "../services/authService";
import { getStudentProfile, registerAdmin, registerStudent } from "../services/studentService";
import "./LoginPage.css";

const LoginPage = () => {
  const navigate = useNavigate();
  const [showIdPrompt, setShowIdPrompt] = useState(false);
  const [studentId, setStudentId] = useState("");
  const [googleUser, setGoogleUser] = useState<{
    name: string;
    email: string;
    photo: string;
  } | null>(null);
  const [isRegistering, setIsRegistering] = useState(false);

  const handleLogin = async () => {
    try {
      const user = await loginWithGoogle();
      const email = user.email || "";
      const profile = email ? await getStudentProfile(email) : { found: false };

      const nextRole = profile.role === "admin" ? "admin" : "student";

      const nextUser = {
        name: user.displayName || "",
        email,
        photo: user.photoURL || "",
      };

      setGoogleUser(nextUser);

      if (nextRole === "admin") {
        const adminRecord = await registerAdmin(
          nextUser.name,
          nextUser.email,
          nextUser.photo
        );
        localStorage.setItem(
          "user",
          JSON.stringify({
            ...nextUser,
            role: adminRecord.role,
            registered: true,
          })
        );
        navigate("/");
        return;
      }

      // Show the student ID prompt
      setShowIdPrompt(true);
    } catch (error) {
      console.error(error);
      alert("Login failed. Please try again.");
    }
  };

  const handleSubmitStudentId = async () => {
    if (!studentId.trim()) {
      alert("Please enter your Student ID.");
      return;
    }

    if (!googleUser) return;

    setIsRegistering(true);
    try {
      // Register/update the student in the backend
      const result = await registerStudent(
        studentId.trim(),
        googleUser.name,
        googleUser.email,
        googleUser.photo
      );

      // Store user data in localStorage
      const userData = {
        name: googleUser.name,
        email: googleUser.email,
        photo: googleUser.photo,
        role: result.role,
        studentId: result.student_id,
        registered: result.registered,
      };

      localStorage.setItem("user", JSON.stringify(userData));
      navigate("/");
    } catch (error) {
      console.error("Registration failed:", error);
      alert("Failed to register student. Please try again.");
    } finally {
      setIsRegistering(false);
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

            {!showIdPrompt ? (
              <>
                <p className="login-description">
                  Sign in using your Google account to simulate Centennial
                  access.
                </p>

                <button className="google-login-btn" onClick={handleLogin}>
                  Sign in with Google
                </button>

                <p className="login-note">
                  Students can access face registration and attendance.
                  <br />
                  Admin and instructors can review and download logs.
                </p>
              </>
            ) : (
              <>
                <p className="login-description">
                  Welcome, <strong>{googleUser?.name}</strong>! Please enter
                  your Student ID to continue.
                </p>

                <input
                  type="text"
                  value={studentId}
                  onChange={(e) => setStudentId(e.target.value)}
                  placeholder="Enter your Student ID (e.g. 301481867)"
                  className="student-id-input"
                  onKeyDown={(e) => e.key === "Enter" && handleSubmitStudentId()}
                />

                <button
                  className="google-login-btn"
                  onClick={handleSubmitStudentId}
                  disabled={isRegistering}
                >
                  {isRegistering ? "Registering..." : "Continue"}
                </button>
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default LoginPage;
