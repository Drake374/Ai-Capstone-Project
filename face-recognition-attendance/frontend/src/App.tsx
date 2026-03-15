import StudentPage from './pages/StudentPage.tsx';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

function App() {

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/student" element={<StudentPage />} />
        {/* other routes */}
      </Routes>
    </BrowserRouter>
  );
}

export default App
