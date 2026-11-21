'use client';
import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [bpm, setBpm] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setBpm(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/process-video', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        if (data.bpm) {
          setBpm(data.bpm);
        } else {
          setError(data.error || 'حدث خطأ أثناء المعالجة');
        }
      } else {
        setError('فشل الاتصال بسيرفر البايثون');
      }
    } catch (err) {
      setError('خطأ في الشبكة: تأكد من أن سيرفر البايثون يعمل');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ padding: '50px', fontFamily: 'Arial', maxWidth: '600px', margin: '0 auto' }}>
      <h1>فحص نبضات القلب عبر الكاميرا (rPPG)</h1>
      <p>ارفع فيديو مدته حوالي دقيقة لتحديد معدل ضربات القلب:</p>

      <form onSubmit={handleSubmit} style={{ marginTop: '20px' }}>
        <input 
          type="file" 
          accept="video/*" 
          onChange={(e) => setFile(e.target.files[0])} 
          style={{ marginBottom: '15px', display: 'block' }}
        />
        <button 
          type="submit" 
          disabled={loading || !file}
          style={{ padding: '10px 20px', background: '#0070f3', color: '#fff', border: 'none', borderRadius: '5px', cursor: 'pointer' }}
        >
          {loading ? 'جاري المعالجة والتحليل...' : 'تحليل الفيديو'}
        </button>
      </form>

      {bpm && (
        <div style={{ marginTop: '20px', padding: '15px', background: '#e6ffed', border: '1px solid #b7eb8f', borderRadius: '5px' }}>
          <h2>معدل نبضات القلب المقدر:</h2>
          <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#389e0d' }}>{bpm} BPM</p>
        </div>
      )}

      {error && (
        <div style={{ marginTop: '20px', padding: '15px', background: '#fff2f0', border: '1px solid #ffccc7', borderRadius: '5px', color: '#ff4d4f' }}>
          <p>{error}</p>
        </div>
      )}
    </main>
  );
}