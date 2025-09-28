generateQRCode() {
                const port = 8080; // You'll need to implement the local server
                const phoneUrl = `http://${this.localIP}:${port}?phone=1`;
                
                const qrcodeContainer = document.getElementById('qrcode');
                qrcodeContainer.innerHTML = ''; // Clear existing content
                
                // 1. Create a canvas element to draw the QR code on
                const canvas = document.createElement('canvas');
                canvas.id = 'qrCanvas';
                qrcodeContainer.appendChild(canvas);

                // 2. Use the canvas element as the target
                QRCode.toCanvas(
                    canvas, // Target the created canvas element
                    phoneUrl,
                    {
                        width: 180,
                        margin: 2,
                        color: {
                            dark: '#333333',
                            light: '#FFFFFF'
                        }
                    },
                    (error) => {
                        if (error) {
                            console.error('QR Code generation failed:', error);
                            // Fallback: show URL as text
                            qrcodeContainer.innerHTML = 
                                `<div style="padding: 20px; background: #ffdddd; color: #cc0000; border-radius: 10px; word-break: break-all;">⚠️ QR Gen Failed. URL: ${phoneUrl}</div>`;
                        }
                    }
                );
            }