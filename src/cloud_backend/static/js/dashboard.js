const API_URL = '/v1/findings';

async function fetchFindings() {
    try {
        const response = await fetch(API_URL);
        if (!response.ok) throw new Error('API request failed');
        const data = await response.json();
        renderTickets(data);
        updateStats(data);
    } catch (error) {
        console.error('Error fetching findings:', error);
        document.getElementById('tickets-container').innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 3rem; color: var(--danger);">
                Error al conectar con la API central.
            </div>`;
    }
}

async function clearTickets() {
    try {
        await fetch(API_URL, { method: 'DELETE' });
        fetchFindings();
    } catch (error) {
        console.error('Error clearing tickets:', error);
    }
}

function normalizeText(text) {
    return (text || '').toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

function getTicketLegend(categoria) {
    const norm = normalizeText(categoria);
    if (norm.includes('apertura')) {
        return 'Se detecta al gerente en hora tardia a su horario normal';
    } else if (norm.includes('vestimenta') || norm.includes('uniforme')) {
        return 'Codigo de vestimenta informal';
    } else if (norm.includes('celular')) {
        return 'Deteccion de celular en horario laboral';
    } else if (norm.includes('audifono')) {
        return 'Deteccion de audifonos';
    } else if (norm.includes('billete') || norm.includes('dinero') || norm.includes('efectivo')) {
        return 'Deteccion de billetes (dinero)';
    } else if (norm.includes('llave')) {
        return 'Deteccion de llaves';
    } else if (norm.includes('alimento') || norm.includes('comida')) {
        return 'Deteccion de consumo de alimentos';
    } else if (norm.includes('pagina') || norm.includes('red') || norm.includes('social') || norm.includes('youtube') || norm.includes('facebook') || norm.includes('tiktok') || norm.includes('instagram')) {
        return 'Ingreso a páginas / redes no autorizadas en monitor';
    } else if (norm.includes('arreglo') || norm.includes('maquillaje') || norm.includes('peinado') || norm.includes('labial')) {
        return 'Actividad de arreglo personal / maquillaje / peinado detectada';
    } else if (norm.includes('saturacion') || norm.includes('fila saturada')) {
        return 'ALERTA: Fila saturada (4+ clientes en espera). Se requiere abrir módulo de apoyo.';
    } else if (norm.includes('abandono')) {
        return 'ALERTA: Abandono de cliente sin recibir atención. Riesgo de insatisfacción.';
    } else if (norm.includes('atencion') || norm.includes('excedid')) {
        return 'ALERTA: Tiempo de atención excedido (>15 min). Posible bloqueo en trámite.';
    }
    return `Incidencia registrada: ${categoria}`;
}

function updateStats(data) {
    document.getElementById('stat-total').innerText = data.length;
    const open = data.filter(t => t.estado === 'DETECTADO').length;
    document.getElementById('stat-open').innerText = open;
    const closed = data.filter(t => t.estado === 'CERRADO').length;
    document.getElementById('stat-closed').innerText = closed;

    // Conteo para la barra multicolor oficial de Izzi
    let cApertura = 0;
    let cVestimenta = 0;
    let cCelulares = 0;
    let cAtipica = 0;
    let cLlaves = 0;
    let cDinero = 0;
    let cAlimentos = 0;
    let cAudifonos = 0;
    let cPaginas = 0;
    let cArreglo = 0;

    data.forEach(t => {
        const norm = normalizeText(t.categoria);
        if (norm.includes('apertura') || norm.includes('horario')) cApertura++;
        else if (norm.includes('vestimenta') || norm.includes('uniforme')) cVestimenta++;
        else if (norm.includes('celular')) cCelulares++;
        else if (norm.includes('atipica') || norm.includes('saturacion') || norm.includes('abandono') || norm.includes('atencion')) cAtipica++;
        else if (norm.includes('llave')) cLlaves++;
        else if (norm.includes('billete') || norm.includes('dinero') || norm.includes('efectivo')) cDinero++;
        else if (norm.includes('alimento') || norm.includes('comida')) cAlimentos++;
        else if (norm.includes('audifono')) cAudifonos++;
        else if (norm.includes('pagina') || norm.includes('red') || norm.includes('social') || norm.includes('youtube') || norm.includes('facebook') || norm.includes('tiktok') || norm.includes('instagram')) cPaginas++;
        else if (norm.includes('arreglo') || norm.includes('maquillaje') || norm.includes('peinado') || norm.includes('labial')) cArreglo++;
    });

    if (document.getElementById('count-apertura')) document.getElementById('count-apertura').innerText = cApertura;
    if (document.getElementById('count-vestimenta')) document.getElementById('count-vestimenta').innerText = cVestimenta;
    if (document.getElementById('count-celulares')) document.getElementById('count-celulares').innerText = cCelulares;
    if (document.getElementById('count-atipica')) document.getElementById('count-atipica').innerText = cAtipica;
    if (document.getElementById('count-llaves')) document.getElementById('count-llaves').innerText = cLlaves;
    if (document.getElementById('count-dinero')) document.getElementById('count-dinero').innerText = cDinero;
    if (document.getElementById('count-alimentos')) document.getElementById('count-alimentos').innerText = cAlimentos;
    if (document.getElementById('count-audifonos')) document.getElementById('count-audifonos').innerText = cAudifonos;
    if (document.getElementById('count-paginas')) document.getElementById('count-paginas').innerText = cPaginas;
    if (document.getElementById('count-arreglo')) document.getElementById('count-arreglo').innerText = cArreglo;
    if (document.getElementById('count-total')) document.getElementById('count-total').innerText = data.length;
}

async function closeTicket(id) {
    try {
        const btn = document.getElementById('btn-' + id);
        btn.innerHTML = '<div class="loader" style="width:16px;height:16px;border-width:2px;"></div>';
        btn.disabled = true;

        const response = await fetch('/v1/findings/' + id + '/status', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'CERRADO' })
        });
        
        if (response.ok) {
            fetchFindings();
        } else {
            alert("Error al cerrar ticket");
            btn.innerHTML = 'Resolver';
            btn.disabled = false;
        }
    } catch (error) {
        console.error('Error updating status:', error);
        alert("Error al cerrar ticket");
    }
}

function parseDateTime(fechaRaw) {
    if (!fechaRaw) return { fecha: 'N/A', hora: 'N/A' };
    if (fechaRaw.includes('T')) {
        const [d, t] = fechaRaw.split('T');
        return { fecha: d, hora: (t || '').split('.')[0] };
    }
    if (fechaRaw.includes(' ')) {
        const [d, t] = fechaRaw.split(' ');
        return { fecha: d, hora: (t || '').split('.')[0] };
    }
    return { fecha: fechaRaw, hora: 'N/A' };
}

function renderTickets(tickets) {
    const container = document.getElementById('tickets-container');
    if (!tickets || tickets.length === 0) {
        container.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 3rem; color: var(--text-muted); background: var(--card-bg); border-radius: 1.2rem; border: 1px solid var(--card-border);">
                No hay hallazgos registrados actualmente.
            </div>`;
        return;
    }

    container.innerHTML = '';
    tickets.forEach(ticket => {
        const isClosed = ticket.estado === 'CERRADO';
        const badgeClass = ticket.estado.toLowerCase();
        const { fecha, hora } = parseDateTime(ticket.fecha_creacion);
        const leyenda = getTicketLegend(ticket.categoria);
        
        const card = document.createElement('div');
        card.className = 'ticket-card';
        card.innerHTML = `
            <div class="ticket-header">
                <span class="ticket-id">#TKT-${ticket.id.toString().padStart(4, '0')}</span>
                <span class="badge ${badgeClass}">${ticket.estado}</span>
            </div>
            
            <div class="ticket-title">${ticket.categoria}</div>

            <div class="ticket-info-box">
                <div class="ticket-info-row">
                    <span class="ticket-info-label">ID:</span>
                    <span class="ticket-info-val">${ticket.id}</span>
                </div>
                <div class="ticket-info-row">
                    <span class="ticket-info-label">Fecha:</span>
                    <span class="ticket-info-val">${fecha}</span>
                </div>
                <div class="ticket-info-row">
                    <span class="ticket-info-label">Hora:</span>
                    <span class="ticket-info-val">${hora}</span>
                </div>
                <div class="ticket-legend-box">
                    <strong>Incidencia:</strong> "${leyenda}"
                </div>
            </div>

            <div class="ticket-meta-footer">
                ${!isClosed && ticket.fecha_vencimiento ? `<div style="color: var(--warning);"><strong>Vencimiento SLA:</strong> ${ticket.fecha_vencimiento}</div>` : ''}
                ${isClosed && ticket.fecha_cierre ? `<div style="color: var(--success);"><strong>Cerrado:</strong> ${ticket.fecha_cierre}</div>` : ''}
            </div>

            <button id="btn-${ticket.id}" 
                    class="btn-resolve" 
                    ${isClosed ? 'disabled' : ''}
                    onclick="closeTicket(${ticket.id})">
                ${isClosed ? 'Resuelto' : 'Marcar como Resuelto'}
            </button>
        `;
        container.appendChild(card);
    });
}

fetchFindings();
setInterval(fetchFindings, 5000);
